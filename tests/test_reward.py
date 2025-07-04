"""
Tests for reward functions
"""

import pytest
from src.grpo.reward import CodeQualityReward, GeneralTextReward


class TestCodeQualityReward:
    def setup_method(self):
        self.reward_func = CodeQualityReward()
    
    def test_good_code_high_reward(self):
        good_code = '''def calculate_factorial(n: int) -> int:
    """Calculate factorial of n.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)'''
        
        reward = self.reward_func.calculate_reward("Write factorial function", good_code)
        assert reward > 0.8
    
    def test_bad_code_low_reward(self):
        bad_code = '''def f(x):
    if x == 0:
        return 1
    else:
        return x * f(x-1)'''
        
        reward = self.reward_func.calculate_reward("Write factorial function", bad_code)
        assert reward < 0.5
    
    def test_syntax_error_zero_reward(self):
        bad_syntax = '''def factorial(n
    return n * factorial(n-1'''
        
        reward = self.reward_func.calculate_reward("Write factorial function", bad_syntax)
        assert reward == 0.0
    
    def test_type_hints_detection(self):
        with_hints = "def add(a: int, b: int) -> int:\n    return a + b"
        without_hints = "def add(a, b):\n    return a + b"
        
        reward_with = self.reward_func.calculate_reward("Add function", with_hints)
        reward_without = self.reward_func.calculate_reward("Add function", without_hints)
        
        assert reward_with > reward_without


class TestGeneralTextReward:
    def setup_method(self):
        self.reward_func = GeneralTextReward()
    
    def test_complete_response(self):
        prompt = "How do I reset a password?"
        good_response = "To reset a password, navigate to the settings menu and click on 'Reset Password'."
        incomplete_response = "To reset a password..."
        
        reward_good = self.reward_func.calculate_reward(prompt, good_response)
        reward_incomplete = self.reward_func.calculate_reward(prompt, incomplete_response)
        
        assert reward_good > reward_incomplete
    
    def test_formatting_check(self):
        prompt = "Explain the process"
        well_formatted = "The process involves three steps. First, you initialize the system."
        poorly_formatted = "the process involves three steps first you initialize the system"
        
        reward_good = self.reward_func.calculate_reward(prompt, well_formatted)
        reward_poor = self.reward_func.calculate_reward(prompt, poorly_formatted)
        
        assert reward_good > reward_poor