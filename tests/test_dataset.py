"""
Tests for dataset functionality
"""

import pytest
import tempfile
import csv
from pathlib import Path
from src.grpo.dataset import GRPODataset


class TestGRPODataset:
    def create_test_csv(self, num_rows=10):
        """Create a temporary CSV file for testing"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        writer = csv.DictWriter(temp_file, fieldnames=['id', 'question', 'answer'])
        writer.writeheader()
        
        for i in range(num_rows):
            writer.writerow({
                'id': str(i),
                'question': f'Question {i}',
                'answer': f'Answer {i}'
            })
        
        temp_file.close()
        return temp_file.name
    
    def test_load_csv_dataset(self):
        csv_path = self.create_test_csv(10)
        
        dataset = GRPODataset(csv_path, train_split=0.8)
        
        assert len(dataset.data) == 10
        assert len(dataset.train_dataset) == 8
        assert len(dataset.eval_dataset) == 2
        
        Path(csv_path).unlink()
    
    def test_dataset_statistics(self):
        csv_path = self.create_test_csv(5)
        
        dataset = GRPODataset(csv_path)
        stats = dataset.get_statistics()
        
        assert stats['total_examples'] == 5
        assert 'avg_question_length' in stats
        assert 'avg_answer_length' in stats
        
        Path(csv_path).unlink()
    
    def test_dataset_validation(self):
        # Create invalid dataset
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        writer = csv.DictWriter(temp_file, fieldnames=['id', 'question', 'answer'])
        writer.writeheader()
        
        # Add row with missing answer
        writer.writerow({
            'id': '1',
            'question': 'Question 1',
            'answer': ''
        })
        
        temp_file.close()
        
        dataset = GRPODataset(temp_file.name)
        is_valid, issues = dataset.validate()
        
        assert not is_valid
        assert len(issues) > 0
        
        Path(temp_file.name).unlink()