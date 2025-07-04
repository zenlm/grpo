#!/usr/bin/env python3
"""
Script to collect code examples from Hanzo Chat for daily fine-tuning
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
import logging
import re
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeExampleCollector:
    """
    Collects code examples from Hanzo Chat conversations
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.hanzo.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def collect_daily_examples(self, date: datetime = None) -> List[Dict[str, Any]]:
        """Collect code examples from a specific day"""
        if date is None:
            date = datetime.now() - timedelta(days=1)  # Yesterday
        
        start_time = date.replace(hour=0, minute=0, second=0)
        end_time = date.replace(hour=23, minute=59, second=59)
        
        logger.info(f"Collecting examples from {start_time} to {end_time}")
        
        # Fetch conversations
        conversations = await self._fetch_conversations(start_time, end_time)
        
        # Extract code examples
        examples = []
        for conv in conversations:
            code_pairs = self._extract_code_pairs(conv)
            examples.extend(code_pairs)
        
        logger.info(f"Collected {len(examples)} code examples")
        return examples
    
    async def _fetch_conversations(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch conversations from Hanzo Chat API"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/v1/conversations"
            params = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "has_code": True  # Only conversations with code
            }
            
            async with session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("conversations", [])
    
    def _extract_code_pairs(self, conversation: Dict) -> List[Dict[str, Any]]:
        """Extract code improvement pairs from conversation"""
        pairs = []
        messages = conversation.get("messages", [])
        
        for i in range(len(messages) - 1):
            msg = messages[i]
            next_msg = messages[i + 1]
            
            # Look for code improvement patterns
            if self._is_code_request(msg) and self._contains_code(next_msg):
                # Extract the original and improved code
                original_code = self._extract_code_blocks(msg.get("content", ""))
                improved_code = self._extract_code_blocks(next_msg.get("content", ""))
                
                if original_code and improved_code:
                    # Determine if the improvement was accepted
                    feedback = self._get_user_feedback(messages, i + 1)
                    
                    if feedback.get("accepted", True):  # Default to accepted
                        pairs.append({
                            "id": f"{conversation['id']}_{i}",
                            "question": msg.get("content", ""),
                            "answer": improved_code[0],  # First code block
                            "rejected_answer": original_code[0] if original_code else "",
                            "context": self._get_context(conversation),
                            "feedback_score": feedback.get("score", 1.0),
                            "timestamp": msg.get("timestamp", "")
                        })
        
        return pairs
    
    def _is_code_request(self, message: Dict) -> bool:
        """Check if message is requesting code or code improvement"""
        content = message.get("content", "").lower()
        patterns = [
            r"improve.*code",
            r"refactor",
            r"optimize",
            r"fix.*bug",
            r"make.*better",
            r"clean.*up",
            r"review.*code",
            r"suggest.*improvement"
        ]
        
        return any(re.search(pattern, content) for pattern in patterns)
    
    def _contains_code(self, message: Dict) -> bool:
        """Check if message contains code"""
        content = message.get("content", "")
        return "```" in content or re.search(r"def\s+\w+|class\s+\w+|import\s+\w+", content)
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from message content"""
        # Extract markdown code blocks
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", content, re.DOTALL)
        
        if not code_blocks:
            # Try to extract code-like content
            lines = content.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if re.match(r"^\s*(def|class|import|from|if|for|while|try)", line):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                    
                    # Check if we've reached the end of a code block
                    if line.strip() == "" and len(code_lines) > 1:
                        code_blocks.append('\n'.join(code_lines[:-1]))
                        code_lines = []
                        in_code = False
            
            if code_lines:
                code_blocks.append('\n'.join(code_lines))
        
        return code_blocks
    
    def _get_user_feedback(self, messages: List[Dict], improved_msg_index: int) -> Dict:
        """Get user feedback on the improved code"""
        feedback = {"accepted": True, "score": 1.0}
        
        # Look for feedback in subsequent messages
        for i in range(improved_msg_index + 1, min(improved_msg_index + 3, len(messages))):
            msg = messages[i]
            content = msg.get("content", "").lower()
            
            # Positive feedback
            if any(word in content for word in ["thanks", "perfect", "great", "excellent", "good"]):
                feedback["score"] = 1.0
            # Negative feedback
            elif any(word in content for word in ["wrong", "incorrect", "bad", "issue", "problem"]):
                feedback["accepted"] = False
                feedback["score"] = 0.3
            # Neutral/modification request
            elif any(word in content for word in ["but", "however", "change", "modify"]):
                feedback["score"] = 0.7
        
        return feedback
    
    def _get_context(self, conversation: Dict) -> Dict[str, Any]:
        """Extract conversation context"""
        return {
            "language": self._detect_language(conversation),
            "project_type": conversation.get("metadata", {}).get("project_type", "unknown"),
            "user_level": conversation.get("metadata", {}).get("user_level", "intermediate")
        }
    
    def _detect_language(self, conversation: Dict) -> str:
        """Detect programming language from conversation"""
        content = " ".join(msg.get("content", "") for msg in conversation.get("messages", []))
        
        # Language detection patterns
        if "python" in content.lower() or "import" in content:
            return "python"
        elif "javascript" in content.lower() or "const" in content or "=>" in content:
            return "javascript"
        elif "java" in content.lower() and "public class" in content:
            return "java"
        elif "typescript" in content.lower() or ": string" in content:
            return "typescript"
        else:
            return "unknown"


class CodeQualityAnalyzer:
    """
    Analyzes code quality to generate training pairs
    """
    
    def __init__(self):
        self.quality_metrics = {
            "has_docstring": 0.2,
            "has_type_hints": 0.2,
            "follows_naming_convention": 0.15,
            "has_error_handling": 0.15,
            "is_efficient": 0.15,
            "is_readable": 0.15
        }
    
    def create_training_pair(self, code: str) -> Tuple[str, str]:
        """Create a bad/good code pair for training"""
        # Analyze current code
        quality_score = self._analyze_quality(code)
        
        if quality_score > 0.8:
            # Code is already good, create a degraded version
            bad_code = self._degrade_code(code)
            return bad_code, code
        else:
            # Code needs improvement, create an improved version
            good_code = self._improve_code(code)
            return code, good_code
    
    def _analyze_quality(self, code: str) -> float:
        """Analyze code quality score"""
        scores = {}
        
        try:
            tree = ast.parse(code)
            
            # Check for docstrings
            scores["has_docstring"] = self._check_docstrings(tree)
            
            # Check for type hints
            scores["has_type_hints"] = self._check_type_hints(tree)
            
            # Check naming conventions
            scores["follows_naming_convention"] = self._check_naming(tree)
            
            # Check error handling
            scores["has_error_handling"] = self._check_error_handling(tree)
            
            # Check efficiency (simplified)
            scores["is_efficient"] = 0.7  # Default
            
            # Check readability
            scores["is_readable"] = self._check_readability(code)
            
        except SyntaxError:
            return 0.0
        
        # Calculate weighted score
        total_score = sum(
            scores.get(metric, 0) * weight 
            for metric, weight in self.quality_metrics.items()
        )
        
        return total_score
    
    def _check_docstrings(self, tree: ast.AST) -> float:
        """Check if functions have docstrings"""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not functions:
            return 0.5
        
        with_docstring = sum(1 for func in functions if ast.get_docstring(func))
        return with_docstring / len(functions)
    
    def _check_type_hints(self, tree: ast.AST) -> float:
        """Check for type hints in functions"""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not functions:
            return 0.5
        
        hints_count = 0
        total_params = 0
        
        for func in functions:
            total_params += len(func.args.args)
            hints_count += sum(1 for arg in func.args.args if arg.annotation)
            if func.returns:
                hints_count += 1
        
        if total_params == 0:
            return 0.5
        
        return hints_count / (total_params + len(functions))
    
    def _check_naming(self, tree: ast.AST) -> float:
        """Check naming conventions"""
        score = 1.0
        
        # Check function names (should be snake_case)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r"^[a-z_][a-z0-9_]*$", node.name):
                    score -= 0.1
            elif isinstance(node, ast.ClassDef):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    score -= 0.1
        
        return max(0, score)
    
    def _check_error_handling(self, tree: ast.AST) -> float:
        """Check for error handling"""
        try_blocks = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        if functions == 0:
            return 0.5
        
        # Expect some error handling
        return min(1.0, try_blocks / (functions * 0.5))
    
    def _check_readability(self, code: str) -> float:
        """Check code readability"""
        lines = code.split('\n')
        
        # Check line length
        long_lines = sum(1 for line in lines if len(line) > 79)
        
        # Check complexity (simplified - count nesting)
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)
        
        score = 1.0
        score -= (long_lines / len(lines)) * 0.3
        score -= max(0, (max_indent - 3) * 0.1)
        
        return max(0, score)
    
    def _improve_code(self, code: str) -> str:
        """Generate improved version of code"""
        # This is a simplified version - in practice, you'd use
        # more sophisticated code transformation
        
        improved = code
        
        # Add docstring if missing
        if 'def ' in code and '"""' not in code:
            improved = re.sub(
                r'(def\s+\w+\([^)]*\):)',
                r'\1\n    """TODO: Add docstring."""',
                improved,
                count=1
            )
        
        # Add basic type hints
        improved = re.sub(
            r'def\s+(\w+)\(([^)]+)\):',
            lambda m: self._add_type_hints(m),
            improved
        )
        
        return improved
    
    def _add_type_hints(self, match) -> str:
        """Add type hints to function definition"""
        func_name = match.group(1)
        params = match.group(2)
        
        # Simple heuristic - add Any type hints
        if ':' not in params:  # No existing type hints
            params_list = [p.strip() for p in params.split(',') if p.strip()]
            typed_params = [f"{p}: Any" for p in params_list]
            return f"def {func_name}({', '.join(typed_params)}) -> Any:"
        
        return match.group(0)
    
    def _degrade_code(self, code: str) -> str:
        """Create a degraded version of good code for training"""
        degraded = code
        
        # Remove docstrings
        degraded = re.sub(r'""".*?"""', '', degraded, flags=re.DOTALL)
        
        # Remove type hints
        degraded = re.sub(r':\s*\w+(?:\[[^\]]+\])?(?=\s*[,)])', '', degraded)
        degraded = re.sub(r'->\s*\w+(?:\[[^\]]+\])?:', ':', degraded)
        
        # Make variable names less descriptive
        degraded = re.sub(r'\b(user_id|user_name|file_path)\b', 'x', degraded)
        
        return degraded


async def main():
    """Main function to collect and process code examples"""
    # Initialize collector
    api_key = os.getenv("HANZO_API_KEY")
    if not api_key:
        logger.error("HANZO_API_KEY environment variable not set")
        return
    
    collector = CodeExampleCollector(api_key)
    analyzer = CodeQualityAnalyzer()
    
    # Collect examples from yesterday
    examples = await collector.collect_daily_examples()
    
    # Process and enhance examples
    enhanced_examples = []
    for example in examples:
        # Create additional training pairs
        code = example.get("answer", "")
        if code:
            bad_code, good_code = analyzer.create_training_pair(code)
            
            enhanced_examples.append({
                "id": example["id"],
                "question": example["question"],
                "answer": good_code,
                "rejected_answer": bad_code,
                "source": "hanzo_chat",
                "timestamp": example["timestamp"],
                "language": example["context"]["language"],
                "feedback_score": example["feedback_score"]
            })
    
    # Save to CSV
    output_dir = Path("data/daily_collections")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"code_examples_{date_str}.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["id", "question", "answer", "rejected_answer", "source", 
                     "timestamp", "language", "feedback_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enhanced_examples)
    
    logger.info(f"Saved {len(enhanced_examples)} examples to {output_file}")
    
    # Optionally trigger fine-tuning
    if len(enhanced_examples) >= 100:  # Minimum threshold
        logger.info("Sufficient examples collected. Triggering fine-tuning...")
        # Call Hanzo ML API to start fine-tuning job
        # This would be implemented based on your ML pipeline


if __name__ == "__main__":
    import os
    asyncio.run(main())