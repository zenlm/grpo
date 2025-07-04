# Comprehensive Detailed Guide to Creating and Using a Skippy Dataset with One-Shot Prompting

This is a detailed guide to creating and using a Skippy dataset with one-shot prompting. It covers the entire process from creating the dataset to using it with a language model.

## 1. Introduction

### 1.1 Purpose

- To provide a complete, unambiguous guide to create a dataset for Skippy and to use advanced prompting techniques (specifically one-shot prompting) with your language model.

### 1.2 Audience

- Beginners or advanced users who require excruciatingly detailed explanations, where even the smallest detail is explicitly defined.

### 1.3 Scope

- This annex covers:
  - CSV dataset structure and examples.
  - Code functions for loading and processing the dataset.
  - Definitions and examples of system prompting vs. one-shot prompting.
  - A detailed, extreme one-shot example (e.g. a "grumpy" response) and guidance on when and how to use each technique.
  - A summary section that recaps every key point.

## 2. Creating a Dataset for Skippy

### 2.1 Dataset Format

- **File Type:** CSV
- **Columns Required:**

  - **id:** Unique identifier (number or string) for each record.
  - **question:** A text field containing the question about the Skippy platform.
  - **answer:** A text field containing the answer that explains the process or feature.

### 2.2 Example CSV (skippy_knowledge_base.csv)

```csv
id,question,answer
1,"How do I set up a new class communication channel in Skippy?","To set up a new class channel, log in to the Skippy admin panel, navigate to the Channels section, click on 'Create New Channel', fill in the required details, and configure notifications and permissions."
2,"How can I reset a teacher's password in Skippy?","Log in to the admin panel, go to the Teachers section, select the teacher, click 'Reset Password', and follow the on-screen instructions."
3,"How do I configure email notifications for parents?","Navigate to the Settings menu, select 'Notifications', choose 'Parent Email Settings', and configure the frequency and types of notifications you want parents to receive."
4,"What's the process for archiving old class data?","Go to the Data Management section, select 'Archive Classes', choose the academic year you want to archive, review the classes to be archived, and click 'Confirm Archive'."
5,"How can I generate student attendance reports?","In the Reports section, select 'Attendance Reports', choose the date range and classes, select the export format (PDF or CSV), and click 'Generate Report'."
6,"How do I add a new teacher to the platform?","Click on 'Users' > 'Teachers' > 'Add New Teacher', enter their details (name, email, subjects), assign them to classes, and send them an invitation email."
7,"What are the steps to create a parent-teacher conference schedule?","Go to 'Events' > 'Conferences', click 'New Conference Schedule', set the date and time slots, assign teachers, and enable parent booking."
8,"How can I monitor student activity on the platform?","Access the 'Analytics' dashboard, select 'Student Activity', choose the time period and metrics you want to view, and export the data if needed."
9,"How do I set up automated grading scales?","Navigate to 'Grading' > 'Grade Scales', click 'Create New Scale', define grade boundaries and point values, and apply it to relevant classes."
10,"What's the procedure for handling data export requests?","Go to 'Privacy & Compliance', select 'Data Requests', review the pending requests, verify the requester's identity, and export the requested data in the specified format."
```

### 2.3 CSV Creation Guidelines

- **Escaping Special Characters:** If your text contains commas, enclose the entire field in double quotes.
- **Newlines:** Avoid newlines within fields unless necessary; if used, enclose the field in double quotes.
- **Consistency:** Ensure all rows have the same number of columns.

## 3. Code for Loading and Processing the Dataset

### 3.1 Python Function to Load CSV

```python
import csv
import json
from typing import List, Dict

def load_skippy_dataset(csv_path: str) -> List[Dict[str, str]]:
    """
    Load the Skippy knowledge base from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing Q&A pairs
    """
    data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append({
                'id': row['id'],
                'question': row['question'].strip(),
                'answer': row['answer'].strip()
            })
    return data

# Example usage
dataset = load_skippy_dataset('skippy_knowledge_base.csv')
print(f"Loaded {len(dataset)} Q&A pairs")
```

### 3.2 Function to Search the Dataset

```python
def search_dataset(dataset: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    """
    Search the dataset for questions containing the query string.
    
    Args:
        dataset (List[Dict[str, str]]): The loaded dataset
        query (str): Search query
        
    Returns:
        List[Dict[str, str]]: Matching Q&A pairs
    """
    query_lower = query.lower()
    matches = []
    
    for item in dataset:
        if query_lower in item['question'].lower():
            matches.append(item)
    
    return matches

# Example usage
results = search_dataset(dataset, "password")
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
```

## 4. Understanding Prompting Techniques

### 4.1 System Prompting

**Definition:** System prompting involves setting the overall behavior, tone, and constraints for the AI model through a system message.

**Example System Prompt:**
```python
system_prompt = """You are a helpful assistant for the Skippy educational platform. 
You provide clear, accurate, and friendly guidance to teachers and administrators 
about using Skippy's features. Always be professional and concise in your responses."""
```

### 4.2 One-Shot Prompting

**Definition:** One-shot prompting provides a single example of the desired input-output format to guide the model's responses.

**Example One-Shot Prompt:**
```python
one_shot_example = """Example:
User: How do I reset a student's password?
Assistant: To reset a student's password in Skippy, follow these steps:
1. Log in to the admin panel
2. Navigate to Users > Students
3. Search for the student by name or ID
4. Click on the student's profile
5. Select "Reset Password" 
6. Choose to either generate a temporary password or send a reset link
7. Confirm the action

The student will receive an email with login instructions."""
```

### 4.3 Extreme One-Shot Example: The Grumpy Assistant

**Purpose:** To demonstrate how one-shot prompting can dramatically change response style.

```python
grumpy_one_shot = """Example:
User: How do I add a new teacher?
Assistant: *sigh* Another person who can't figure out the obvious... Fine, I'll spell it out for you:
1. Click the 'Users' button - it's right there on the left, can't miss it
2. Hit 'Teachers' - shocking, I know
3. There's a big 'Add New Teacher' button - click it
4. Fill in their information - name, email, you know, the basics
5. Assign them to classes - unless you want them doing nothing
6. Send the invitation - and we're done here

Was that really so hard to figure out on your own? The interface practically screams the steps at you..."""
```

## 5. Implementing Different Prompting Strategies

### 5.1 Basic Implementation

```python
from typing import Optional

class SkippyAssistant:
    def __init__(self, dataset: List[Dict[str, str]], 
                 system_prompt: str,
                 one_shot_example: Optional[str] = None):
        self.dataset = dataset
        self.system_prompt = system_prompt
        self.one_shot_example = one_shot_example
    
    def format_prompt(self, user_question: str) -> str:
        """Format the prompt with system message and one-shot example."""
        prompt_parts = [self.system_prompt]
        
        if self.one_shot_example:
            prompt_parts.append(f"\n{self.one_shot_example}")
        
        prompt_parts.append(f"\nUser: {user_question}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def search_knowledge_base(self, query: str) -> Optional[str]:
        """Search the dataset for relevant answers."""
        matches = search_dataset(self.dataset, query)
        if matches:
            return matches[0]['answer']
        return None
```

### 5.2 Usage Examples

```python
# Standard helpful assistant
standard_assistant = SkippyAssistant(
    dataset=dataset,
    system_prompt="You are a helpful Skippy platform assistant.",
    one_shot_example=one_shot_example
)

# Grumpy assistant (not recommended for production!)
grumpy_assistant = SkippyAssistant(
    dataset=dataset,
    system_prompt="You are a grumpy but knowledgeable Skippy platform assistant.",
    one_shot_example=grumpy_one_shot
)
```

## 6. Best Practices and Guidelines

### 6.1 When to Use System Prompting

- **Setting Overall Behavior:** Use system prompts to establish the assistant's personality, expertise level, and communication style.
- **Defining Constraints:** Specify what the assistant should and shouldn't do.
- **Establishing Context:** Provide background information about Skippy that applies to all interactions.

### 6.2 When to Use One-Shot Prompting

- **Format Specification:** When you need responses in a specific format.
- **Style Demonstration:** To show the exact tone and structure you want.
- **Complex Examples:** When the desired behavior is easier to show than describe.

### 6.3 When NOT to Use Extreme Styles

- **Professional Settings:** Never use grumpy or unprofessional tones in production.
- **User Trust:** Maintain a helpful, respectful tone to build user confidence.
- **Brand Consistency:** Ensure the assistant's style aligns with Skippy's brand values.

## 7. Integration with Language Models

### 7.1 OpenAI API Example

```python
import openai

def get_skippy_response(user_question: str, assistant: SkippyAssistant) -> str:
    """Get a response using OpenAI's API."""
    # First, try to find answer in knowledge base
    kb_answer = assistant.search_knowledge_base(user_question)
    
    if kb_answer:
        return kb_answer
    
    # If not found, use LLM with prompting
    prompt = assistant.format_prompt(user_question)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": assistant.system_prompt},
            {"role": "user", "content": user_question}
        ]
    )
    
    return response['choices'][0]['message']['content']
```

### 7.2 Local Model Integration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class LocalSkippyAssistant(SkippyAssistant):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_response(self, user_question: str) -> str:
        """Generate response using local model."""
        prompt = self.format_prompt(user_question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()
```

## 8. Testing and Validation

### 8.1 Test Suite

```python
def test_skippy_assistant():
    """Test the assistant with various questions."""
    test_questions = [
        "How do I reset a teacher's password?",
        "What's the process for creating a new class?",
        "How can I generate attendance reports?",
        "Help me set up parent notifications"
    ]
    
    assistant = SkippyAssistant(
        dataset=dataset,
        system_prompt="You are a helpful Skippy assistant.",
        one_shot_example=one_shot_example
    )
    
    for question in test_questions:
        print(f"Q: {question}")
        response = assistant.search_knowledge_base(question)
        if response:
            print(f"A (from KB): {response}")
        else:
            print("A: [Would use LLM here]")
        print("-" * 50)
```

### 8.2 Quality Metrics

```python
def evaluate_responses(assistant: SkippyAssistant, test_set: List[Dict[str, str]]):
    """Evaluate response quality."""
    metrics = {
        'found_in_kb': 0,
        'not_found': 0,
        'total': len(test_set)
    }
    
    for test_case in test_set:
        response = assistant.search_knowledge_base(test_case['question'])
        if response:
            metrics['found_in_kb'] += 1
        else:
            metrics['not_found'] += 1
    
    print(f"Knowledge Base Coverage: {metrics['found_in_kb']}/{metrics['total']} "
          f"({metrics['found_in_kb']/metrics['total']*100:.1f}%)")
```

## 9. Advanced Techniques

### 9.1 Multi-Shot Prompting

```python
multi_shot_examples = """Example 1:
User: How do I add a student?
Assistant: To add a student, go to Users > Students > Add New Student, fill in their details, and click Save.

Example 2:
User: Where can I find attendance reports?
Assistant: You can find attendance reports in the Reports section. Select 'Attendance Reports', choose your parameters, and generate.

Example 3:
User: How do I change notification settings?
Assistant: Navigate to Settings > Notifications, where you can configure email, SMS, and in-app notification preferences."""
```

### 9.2 Chain of Thought Prompting

```python
cot_system_prompt = """You are a Skippy platform assistant. When answering questions:
1. First, identify what the user is trying to accomplish
2. Then, break down the steps needed
3. Finally, provide clear, numbered instructions

Always think through the user's goal before responding."""
```

### 9.3 Dynamic Prompting

```python
def create_dynamic_prompt(user_role: str, urgency: str) -> str:
    """Create prompts based on context."""
    base_prompt = "You are a Skippy platform assistant"
    
    role_modifiers = {
        'teacher': "speaking to a teacher who may not be tech-savvy",
        'admin': "speaking to a system administrator who needs detailed information",
        'parent': "speaking to a parent who needs simple, clear instructions"
    }
    
    urgency_modifiers = {
        'high': ". Provide the fastest solution possible.",
        'normal': ". Provide clear, thorough instructions.",
        'low': ". Take time to explain the why behind each step."
    }
    
    return f"{base_prompt} {role_modifiers.get(user_role, '')} {urgency_modifiers.get(urgency, '')}"
```

## 10. Production Deployment

### 10.1 Error Handling

```python
class ProductionSkippyAssistant(SkippyAssistant):
    def get_response(self, user_question: str) -> str:
        """Get response with error handling."""
        try:
            # Try knowledge base first
            kb_response = self.search_knowledge_base(user_question)
            if kb_response:
                return kb_response
            
            # Fallback to LLM
            return self.get_llm_response(user_question)
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return "I apologize, but I'm having trouble accessing the information. Please try again or contact support."
    
    def validate_response(self, response: str) -> str:
        """Validate and sanitize responses."""
        # Remove any sensitive information patterns
        response = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REMOVED]', response)
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', response)
        
        # Ensure response isn't too long
        if len(response) > 1000:
            response = response[:997] + "..."
        
        return response
```

### 10.2 Logging and Monitoring

```python
import logging
from datetime import datetime

class MonitoredAssistant(ProductionSkippyAssistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger('skippy_assistant')
        self.metrics = {
            'total_queries': 0,
            'kb_hits': 0,
            'llm_fallbacks': 0,
            'errors': 0
        }
    
    def get_response(self, user_question: str) -> str:
        """Get response with monitoring."""
        start_time = datetime.now()
        self.metrics['total_queries'] += 1
        
        try:
            response = super().get_response(user_question)
            
            # Log query and response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Query processed in {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"Error processing query: {e}")
            raise
```

## 11. Summary and Key Takeaways

### 11.1 Dataset Creation Summary

1. **CSV Format:** Use id, question, answer columns
2. **Escaping:** Use quotes for fields containing commas
3. **Consistency:** Ensure all rows have the same structure
4. **Quality:** Write clear, comprehensive answers

### 11.2 Prompting Techniques Summary

1. **System Prompting:** Sets overall behavior and constraints
2. **One-Shot Prompting:** Provides format and style examples
3. **Multi-Shot Prompting:** Multiple examples for complex patterns
4. **Dynamic Prompting:** Adapts based on context

### 11.3 Implementation Summary

1. **Load Dataset:** Use CSV reader for simple loading
2. **Search Function:** Implement basic keyword matching
3. **Prompt Formatting:** Combine system and one-shot prompts
4. **Error Handling:** Always include fallbacks and validation

### 11.4 Best Practices Summary

1. **Professional Tone:** Always maintain respectful communication
2. **Knowledge Base First:** Check dataset before using LLM
3. **Monitor Performance:** Track metrics and response times
4. **Validate Output:** Sanitize responses for sensitive data

### 11.5 When to Use Each Technique

| Technique | Use Case | Example |
|-----------|----------|---------|
| System Prompting | Setting personality | "You are a friendly assistant" |
| One-Shot | Format specification | Show exact response structure |
| Multi-Shot | Complex patterns | Multiple Q&A examples |
| Dynamic | Context-aware | Adjust based on user role |

## 12. Conclusion

This guide has covered every aspect of creating and using a Skippy dataset with various prompting techniques. Remember:

1. **Start Simple:** Begin with basic system prompting and a clean dataset
2. **Test Thoroughly:** Validate responses before production deployment
3. **Monitor Continuously:** Track performance and user satisfaction
4. **Iterate:** Improve your dataset and prompts based on user feedback

The key to success is maintaining a balance between comprehensive knowledge base coverage and intelligent LLM fallbacks, while always keeping the user experience professional and helpful.

## Appendix A: Complete Code Example

```python
"""
Complete Skippy Assistant Implementation
"""

import csv
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('skippy_assistant')

class SkippyAssistant:
    """Main Skippy Assistant class with all features."""
    
    def __init__(self, csv_path: str, system_prompt: str, 
                 one_shot_example: Optional[str] = None):
        self.dataset = self.load_dataset(csv_path)
        self.system_prompt = system_prompt
        self.one_shot_example = one_shot_example
        self.metrics = {
            'total_queries': 0,
            'kb_hits': 0,
            'llm_fallbacks': 0,
            'errors': 0
        }
    
    def load_dataset(self, csv_path: str) -> List[Dict[str, str]]:
        """Load the Skippy knowledge base from CSV."""
        data = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    data.append({
                        'id': row['id'],
                        'question': row['question'].strip(),
                        'answer': row['answer'].strip()
                    })
            logger.info(f"Loaded {len(data)} Q&A pairs from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        return data
    
    def search_knowledge_base(self, query: str) -> Optional[str]:
        """Search the dataset for relevant answers."""
        query_lower = query.lower()
        
        # Exact match first
        for item in self.dataset:
            if query_lower == item['question'].lower():
                self.metrics['kb_hits'] += 1
                return item['answer']
        
        # Partial match
        for item in self.dataset:
            if query_lower in item['question'].lower():
                self.metrics['kb_hits'] += 1
                return item['answer']
        
        # Keyword match
        keywords = query_lower.split()
        best_match = None
        best_score = 0
        
        for item in self.dataset:
            question_lower = item['question'].lower()
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_match and best_score >= len(keywords) * 0.5:
            self.metrics['kb_hits'] += 1
            return best_match['answer']
        
        return None
    
    def format_prompt(self, user_question: str) -> str:
        """Format the prompt with system message and examples."""
        prompt_parts = [self.system_prompt]
        
        if self.one_shot_example:
            prompt_parts.append(f"\n{self.one_shot_example}")
        
        prompt_parts.append(f"\nUser: {user_question}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def validate_response(self, response: str) -> str:
        """Validate and sanitize responses."""
        # Remove sensitive information
        response = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REMOVED]', response)
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                         '[EMAIL]', response)
        
        # Ensure reasonable length
        if len(response) > 1000:
            response = response[:997] + "..."
        
        return response
    
    def get_response(self, user_question: str) -> str:
        """Get response with full error handling and monitoring."""
        start_time = datetime.now()
        self.metrics['total_queries'] += 1
        
        try:
            # Try knowledge base first
            kb_response = self.search_knowledge_base(user_question)
            if kb_response:
                response = kb_response
                source = "knowledge_base"
            else:
                # In production, this would call your LLM
                self.metrics['llm_fallbacks'] += 1
                response = "I don't have specific information about that in my knowledge base. Please contact support for assistance."
                source = "llm_fallback"
            
            # Validate response
            response = self.validate_response(response)
            
            # Log performance
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query processed in {response_time:.2f}s from {source}")
            
            return response
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error. Please try again or contact support."
    
    def get_metrics(self) -> Dict[str, any]:
        """Return current metrics."""
        if self.metrics['total_queries'] > 0:
            hit_rate = self.metrics['kb_hits'] / self.metrics['total_queries'] * 100
        else:
            hit_rate = 0
        
        return {
            **self.metrics,
            'kb_hit_rate': f"{hit_rate:.1f}%"
        }

# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = SkippyAssistant(
        csv_path='skippy_knowledge_base.csv',
        system_prompt="You are a helpful assistant for the Skippy educational platform.",
        one_shot_example="""Example:
User: How do I reset a password?
Assistant: To reset a password, navigate to the Users section, find the specific user, 
and click the 'Reset Password' button. You can then either generate a temporary password 
or send a reset link to their email."""
    )
    
    # Test queries
    test_queries = [
        "How do I reset a teacher's password?",
        "How can I create attendance reports?",
        "What's the process for adding new students?",
        "How do I configure notifications?"
    ]
    
    print("Skippy Assistant Test Run")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = assistant.get_response(query)
        print(f"Assistant: {response}")
    
    print("\n" + "=" * 50)
    print("Metrics:", assistant.get_metrics())
```

This comprehensive guide provides everything needed to create, implement, and deploy a Skippy assistant with various prompting techniques. The modular design allows for easy customization and extension based on specific needs.