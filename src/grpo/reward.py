"""
Reward functions for GRPO training
"""

import ast
import re
from typing import Dict, List, Any, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Base class for reward functions"""
    
    @abstractmethod
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward for a response"""
        pass


class CodeQualityReward(RewardFunction):
    """
    Reward function for code quality assessment
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "syntax_correctness": 0.2,
            "documentation": 0.2,
            "type_hints": 0.15,
            "error_handling": 0.15,
            "code_style": 0.15,
            "efficiency": 0.15
        }
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward based on code quality metrics"""
        scores = {
            "syntax_correctness": self._check_syntax(response),
            "documentation": self._check_documentation(response),
            "type_hints": self._check_type_hints(response),
            "error_handling": self._check_error_handling(response),
            "code_style": self._check_code_style(response),
            "efficiency": self._check_efficiency(response)
        }
        
        # Weighted average
        total_reward = sum(
            scores[metric] * self.weights[metric] 
            for metric in scores
        )
        
        return total_reward
    
    def _check_syntax(self, code: str) -> float:
        """Check if code has valid syntax"""
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0
    
    def _check_documentation(self, code: str) -> float:
        """Check documentation quality"""
        doc_patterns = [
            r'""".*?"""',  # Docstrings
            r"'''.*?'''",  # Alternative docstrings
            r'#.*',        # Comments
        ]
        
        doc_score = 0.0
        
        # Check for docstrings
        if re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL):
            doc_score += 0.5
        
        # Check for inline comments
        comment_lines = len(re.findall(r'#.*', code))
        total_lines = len(code.split('\n'))
        
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            doc_score += min(0.5, comment_ratio * 2)  # Cap at 0.5
        
        return doc_score
    
    def _check_type_hints(self, code: str) -> float:
        """Check for type hints"""
        type_hint_patterns = [
            r'def\s+\w+\s*\([^)]*:\s*\w+',  # Function parameter hints
            r'->\s*\w+',                      # Return type hints
            r':\s*(?:List|Dict|Tuple|Optional|Union|Any)',  # Complex types
        ]
        
        hints_found = 0
        for pattern in type_hint_patterns:
            hints_found += len(re.findall(pattern, code))
        
        # Check function definitions
        func_defs = len(re.findall(r'def\s+\w+', code))
        
        if func_defs > 0:
            # Expect at least one type hint per function
            return min(1.0, hints_found / (func_defs * 2))
        
        return 0.5  # No functions, neutral score
    
    def _check_error_handling(self, code: str) -> float:
        """Check for proper error handling"""
        error_patterns = [
            r'try:',
            r'except\s+\w+',
            r'raise\s+\w+',
            r'if\s+.*is\s+None',
            r'if\s+not\s+\w+:',
        ]
        
        error_score = 0.0
        
        # Check for try-except blocks
        if 'try:' in code and 'except' in code:
            error_score += 0.4
        
        # Check for specific exception handling
        if re.search(r'except\s+\w+Error', code):
            error_score += 0.2
        
        # Check for validation
        if re.search(r'if\s+.*(<|>|==|!=|is)', code):
            error_score += 0.2
        
        # Check for raising exceptions
        if 'raise' in code:
            error_score += 0.2
        
        return min(1.0, error_score)
    
    def _check_code_style(self, code: str) -> float:
        """Check code style and formatting"""
        style_score = 1.0
        
        # Check line length
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 79)
        if long_lines > 0:
            style_score -= (long_lines / len(lines)) * 0.3
        
        # Check naming conventions
        # Snake case for functions
        if re.search(r'def\s+[A-Z]', code):  # CamelCase function
            style_score -= 0.2
        
        # Check for proper spacing
        if re.search(r'[=+\-*/](?=[^\s=])|(?<=[^\s=])[=+\-*/]', code):
            style_score -= 0.1
        
        # Check indentation consistency
        indent_sizes = set()
        for line in lines:
            if line and line[0] == ' ':
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indent_sizes.add(indent)
        
        if len(indent_sizes) > 2:  # Inconsistent indentation
            style_score -= 0.2
        
        return max(0.0, style_score)
    
    def _check_efficiency(self, code: str) -> float:
        """Check for code efficiency patterns"""
        efficiency_score = 0.7  # Base score
        
        # Good patterns
        good_patterns = [
            r'\.join\(',           # Using join instead of concatenation
            r'with\s+',            # Context managers
            r'@\w+',               # Decorators
            r'yield\s+',           # Generators
            r'[a-z_]+\s+comprehension',  # Comprehensions (heuristic)
        ]
        
        # Bad patterns
        bad_patterns = [
            r'for.*:\s*\n\s*.*\.append\(',  # Appending in loop
            r'\+\s*=.*\+\s*=.*\+\s*=',      # Multiple string concatenations
            r'global\s+',                     # Global variables
        ]
        
        # Check for good patterns
        for pattern in good_patterns:
            if re.search(pattern, code):
                efficiency_score += 0.1
        
        # Check for bad patterns
        for pattern in bad_patterns:
            if re.search(pattern, code):
                efficiency_score -= 0.15
        
        return max(0.0, min(1.0, efficiency_score))


class GeneralTextReward(RewardFunction):
    """
    General reward function for text quality
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "correctness": 0.4,
            "formatting": 0.3,
            "completeness": 0.3
        }
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward for general text responses"""
        scores = {
            "correctness": self._check_correctness(prompt, response),
            "formatting": self._check_formatting(response),
            "completeness": self._check_completeness(prompt, response)
        }
        
        total_reward = sum(
            scores[metric] * self.weights[metric] 
            for metric in scores
        )
        
        return total_reward
    
    def _check_correctness(self, prompt: str, response: str) -> float:
        """Check if response addresses the prompt"""
        # Simple heuristic: check for keywords from prompt in response
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'or'}
        prompt_words -= common_words
        response_words -= common_words
        
        if not prompt_words:
            return 0.5
        
        overlap = len(prompt_words & response_words) / len(prompt_words)
        return min(1.0, overlap * 1.5)  # Boost score slightly
    
    def _check_formatting(self, response: str) -> float:
        """Check response formatting"""
        format_score = 1.0
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[0].isupper():
                format_score -= 0.1
        
        # Check for punctuation
        if not re.search(r'[.!?]$', response.strip()):
            format_score -= 0.2
        
        # Check for reasonable length
        if len(response) < 10:
            format_score -= 0.3
        
        return max(0.0, format_score)
    
    def _check_completeness(self, prompt: str, response: str) -> float:
        """Check if response is complete"""
        # Check for incomplete sentences
        if response.strip().endswith(('...', '..', '-')):
            return 0.3
        
        # Check for minimum length based on prompt
        expected_length = len(prompt) * 2  # Heuristic
        actual_length = len(response)
        
        if actual_length < expected_length * 0.5:
            return 0.5
        elif actual_length < expected_length:
            return 0.8
        else:
            return 1.0


class CompositeReward(RewardFunction):
    """
    Composite reward function that combines multiple reward functions
    """
    
    def __init__(self, reward_functions: List[Tuple[RewardFunction, float]]):
        """
        Args:
            reward_functions: List of (reward_function, weight) tuples
        """
        self.reward_functions = reward_functions
        total_weight = sum(weight for _, weight in reward_functions)
        
        # Normalize weights
        self.reward_functions = [
            (func, weight / total_weight) 
            for func, weight in reward_functions
        ]
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate composite reward"""
        total_reward = 0.0
        
        for func, weight in self.reward_functions:
            try:
                reward = func.calculate_reward(prompt, response)
                total_reward += reward * weight
            except Exception as e:
                logger.warning(f"Error in reward function {func.__class__.__name__}: {e}")
                # Continue with other functions
        
        return total_reward


class AdaptiveReward(RewardFunction):
    """
    Adaptive reward function that learns from feedback
    """
    
    def __init__(self, base_reward: RewardFunction):
        self.base_reward = base_reward
        self.feedback_history: List[Dict[str, Any]] = []
        self.adjustment_factor = 0.0
    
    def calculate_reward(self, prompt: str, response: str) -> float:
        """Calculate reward with adaptive adjustments"""
        base_score = self.base_reward.calculate_reward(prompt, response)
        
        # Apply learned adjustments
        adjusted_score = base_score * (1 + self.adjustment_factor)
        
        return max(0.0, min(1.0, adjusted_score))
    
    def add_feedback(self, prompt: str, response: str, human_score: float):
        """Add human feedback to improve reward function"""
        ai_score = self.base_reward.calculate_reward(prompt, response)
        
        self.feedback_history.append({
            "prompt": prompt,
            "response": response,
            "ai_score": ai_score,
            "human_score": human_score,
            "difference": human_score - ai_score
        })
        
        # Update adjustment factor based on recent feedback
        if len(self.feedback_history) >= 10:
            recent_diffs = [
                fb["difference"] 
                for fb in self.feedback_history[-10:]
            ]
            self.adjustment_factor = sum(recent_diffs) / len(recent_diffs)
            
            logger.info(f"Updated adjustment factor: {self.adjustment_factor}")