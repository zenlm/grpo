"""
GRPO Trainer implementation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    Guided Reinforcement Policy Optimization Trainer
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        output_dir: str = "./outputs"
    ):
        self.model_name = model_name
        self.config = config
        self.output_dir = output_dir
        
        # Initialize model and tokenizer
        self._setup_model()
        
    def _setup_model(self):
        """Initialize model with LoRA configuration"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_config = self.config.get("model", {})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=model_config.get("device_map", "auto"),
            torch_dtype=model_config.get("torch_dtype", "auto"),
            load_in_4bit=model_config.get("load_in_4bit", True)
        )
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config["lora"]["target_modules"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info("Model setup complete")
    
    def train(self, dataset):
        """Train the model using GRPO"""
        training_config = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.config["training"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            max_steps=self.config["training"]["max_steps"],
            fp16=self.config["training"]["fp16"],
            push_to_hub=self.config["training"]["push_to_hub"],
            report_to=self.config["training"]["report_to"]
        )
        
        trainer = DPOTrainer(
            model=self.model,
            args=training_config,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset,
            tokenizer=self.tokenizer,
            max_length=self.config["dataset"]["max_length"],
            max_prompt_length=self.config["dataset"]["max_length"] // 2,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")
        
        return trainer
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the trained model"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        generation_config = self.config.get("generation", {})
        generation_config.update(kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=generation_config.get("max_new_tokens", 256),
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.9),
                do_sample=generation_config.get("do_sample", True),
                num_return_sequences=generation_config.get("num_return_sequences", 1)
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split(prompt)[-1].strip()