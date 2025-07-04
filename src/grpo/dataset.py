"""
GRPO Dataset implementation for loading and processing training data
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GRPODataset:
    """
    Dataset class for GRPO training
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 512,
        train_split: float = 0.9,
        tokenizer=None,
        format_type: str = "csv"
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.train_split = train_split
        self.tokenizer = tokenizer
        self.format_type = format_type
        
        # Load data
        self.data = self._load_data()
        
        # Create train/eval splits
        self.train_dataset, self.eval_dataset = self._create_splits()
        
    def _load_data(self) -> List[Dict[str, str]]:
        """Load data from file"""
        logger.info(f"Loading data from {self.data_path}")
        
        if self.format_type == "csv":
            return self._load_csv()
        elif self.format_type == "json":
            return self._load_json()
        elif self.format_type == "jsonl":
            return self._load_jsonl()
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")
    
    def _load_csv(self) -> List[Dict[str, str]]:
        """Load data from CSV file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'id': row.get('id', str(len(data))),
                    'question': row['question'].strip(),
                    'answer': row['answer'].strip()
                })
        return data
    
    def _load_json(self) -> List[Dict[str, str]]:
        """Load data from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure proper format
        if isinstance(data, dict):
            data = data.get('data', [])
        
        return data
    
    def _load_jsonl(self) -> List[Dict[str, str]]:
        """Load data from JSONL file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _create_splits(self) -> Tuple[HFDataset, HFDataset]:
        """Create train/eval splits"""
        logger.info(f"Creating train/eval splits with ratio {self.train_split}")
        
        # Convert to pandas for easy splitting
        df = pd.DataFrame(self.data)
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        train_size = int(len(df) * self.train_split)
        train_df = df[:train_size]
        eval_df = df[train_size:]
        
        logger.info(f"Train size: {len(train_df)}, Eval size: {len(eval_df)}")
        
        # Convert to HuggingFace datasets
        train_dataset = HFDataset.from_pandas(train_df)
        eval_dataset = HFDataset.from_pandas(eval_df)
        
        # Process datasets
        if self.tokenizer:
            train_dataset = train_dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            eval_dataset = eval_dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        return train_dataset, eval_dataset
    
    def _preprocess_function(self, examples):
        """Preprocess examples for training"""
        # Format prompts
        prompts = []
        responses = []
        
        for question, answer in zip(examples['question'], examples['answer']):
            prompt = f"Question: {question}\nAnswer:"
            prompts.append(prompt)
            responses.append(answer)
        
        # Tokenize
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize responses
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                responses,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def format_for_grpo(self) -> DatasetDict:
        """Format dataset for GRPO training"""
        # GRPO requires specific format with chosen/rejected responses
        formatted_data = []
        
        for item in self.data:
            # For GRPO, we need to create preference pairs
            # This is a simplified version - in practice, you'd have actual preferences
            formatted_item = {
                "prompt": f"Question: {item['question']}\nAnswer:",
                "chosen": item['answer'],
                "rejected": self._generate_rejected_response(item['answer'])
            }
            formatted_data.append(formatted_item)
        
        # Create dataset
        dataset = HFDataset.from_list(formatted_data)
        
        # Split
        train_size = int(len(dataset) * self.train_split)
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })
    
    def _generate_rejected_response(self, chosen: str) -> str:
        """Generate a rejected response for GRPO training"""
        # In practice, this would be a lower-quality response
        # For demo purposes, we'll create a simplified version
        if len(chosen) > 50:
            return chosen[:30] + "..."  # Truncated response
        else:
            return "I don't know."  # Generic unhelpful response
    
    def save_processed(self, output_path: Union[str, Path]):
        """Save processed dataset"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        train_path = output_path / "train.jsonl"
        eval_path = output_path / "eval.jsonl"
        
        self._save_jsonl(self.train_dataset, train_path)
        self._save_jsonl(self.eval_dataset, eval_path)
        
        logger.info(f"Saved processed dataset to {output_path}")
    
    def _save_jsonl(self, dataset: HFDataset, path: Path):
        """Save dataset as JSONL"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
    
    def get_statistics(self) -> Dict[str, any]:
        """Get dataset statistics"""
        stats = {
            "total_examples": len(self.data),
            "train_examples": len(self.train_dataset),
            "eval_examples": len(self.eval_dataset),
            "avg_question_length": sum(len(d['question']) for d in self.data) / len(self.data),
            "avg_answer_length": sum(len(d['answer']) for d in self.data) / len(self.data),
        }
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate dataset integrity"""
        issues = []
        
        # Check required fields
        for i, item in enumerate(self.data):
            if 'question' not in item or not item['question']:
                issues.append(f"Row {i}: Missing or empty question")
            if 'answer' not in item or not item['answer']:
                issues.append(f"Row {i}: Missing or empty answer")
        
        # Check for duplicates
        questions = [d['question'] for d in self.data]
        if len(questions) != len(set(questions)):
            issues.append("Duplicate questions found in dataset")
        
        # Check lengths
        for i, item in enumerate(self.data):
            if len(item['question']) > 1000:
                issues.append(f"Row {i}: Question too long (>1000 chars)")
            if len(item['answer']) > 2000:
                issues.append(f"Row {i}: Answer too long (>2000 chars)")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class StreamingGRPODataset(GRPODataset):
    """
    Streaming version of GRPO dataset for large files
    """
    
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 1000)
        super().__init__(*args, **kwargs)
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load data in streaming fashion"""
        # For streaming, we don't load all data at once
        # Instead, we'll yield batches
        return []
    
    def stream_batches(self):
        """Stream data in batches"""
        if self.format_type == "csv":
            yield from self._stream_csv()
        elif self.format_type == "jsonl":
            yield from self._stream_jsonl()
        else:
            raise ValueError(f"Streaming not supported for {self.format_type}")
    
    def _stream_csv(self):
        """Stream CSV data in batches"""
        batch = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch.append({
                    'question': row['question'].strip(),
                    'answer': row['answer'].strip()
                })
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            
            if batch:
                yield batch
    
    def _stream_jsonl(self):
        """Stream JSONL data in batches"""
        batch = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line))
                    
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
            
            if batch:
                yield batch