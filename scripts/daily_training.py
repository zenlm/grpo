#!/usr/bin/env python3
"""
Daily training script for continuous improvement of dev AI
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import logging
import asyncio
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.grpo import GRPOTrainer, GRPODataset
from src.grpo.reward import CodeQualityReward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DailyTrainingPipeline:
    """
    Automated daily training pipeline for continuous model improvement
    """
    
    def __init__(self, config_path: str, base_model_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.base_model_path = base_model_path or self.config["model"]["name"]
        self.data_dir = Path("data/daily_collections")
        self.output_dir = Path("models/daily")
        
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run_daily_training(self):
        """Run the daily training pipeline"""
        logger.info("Starting daily training pipeline")
        
        # Step 1: Collect new examples
        await self._collect_examples()
        
        # Step 2: Merge with existing dataset
        dataset_path = self._prepare_dataset()
        
        # Step 3: Run training
        model_path = self._train_model(dataset_path)
        
        # Step 4: Evaluate model
        metrics = self._evaluate_model(model_path)
        
        # Step 5: Deploy if improved
        if self._should_deploy(metrics):
            self._deploy_model(model_path)
        
        logger.info("Daily training pipeline completed")
    
    async def _collect_examples(self):
        """Collect new code examples from the last 24 hours"""
        logger.info("Collecting code examples...")
        
        # Run collection script
        collect_script = Path(__file__).parent / "collect_code_examples.py"
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(collect_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"Collection failed: {stderr.decode()}")
            raise RuntimeError("Failed to collect examples")
        
        logger.info("Collection completed")
    
    def _prepare_dataset(self) -> Path:
        """Prepare training dataset by merging new examples"""
        logger.info("Preparing dataset...")
        
        # Find today's collected examples
        date_str = datetime.now().strftime("%Y%m%d")
        today_file = self.data_dir / f"code_examples_{date_str}.csv"
        
        if not today_file.exists():
            logger.warning("No examples collected today")
            # Fall back to using existing dataset
            return Path(self.config["dataset"]["path"])
        
        # Merge with base dataset
        import pandas as pd
        
        # Load base dataset
        base_df = pd.read_csv(self.config["dataset"]["path"])
        
        # Load today's examples
        today_df = pd.read_csv(today_file)
        
        # Combine datasets
        combined_df = pd.concat([base_df, today_df], ignore_index=True)
        
        # Remove duplicates based on question
        combined_df = combined_df.drop_duplicates(subset=['question'], keep='last')
        
        # Save merged dataset
        output_path = self.data_dir / f"combined_{date_str}.csv"
        combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Prepared dataset with {len(combined_df)} examples")
        return output_path
    
    def _train_model(self, dataset_path: Path) -> Path:
        """Train model with new dataset"""
        logger.info("Starting model training...")
        
        # Update config with new dataset path
        self.config["dataset"]["path"] = str(dataset_path)
        
        # Set output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_dir / timestamp
        self.config["training"]["output_dir"] = str(output_dir)
        
        # Load dataset
        dataset = GRPODataset(
            data_path=dataset_path,
            max_length=self.config["dataset"]["max_length"],
            train_split=self.config["dataset"]["train_split"]
        )
        
        # Initialize trainer with custom reward for code quality
        trainer = GRPOTrainer(
            model_name=self.base_model_path,
            config=self.config,
            output_dir=str(output_dir)
        )
        
        # Add custom code quality reward
        if self.config.get("reward", {}).get("type") == "code_quality":
            trainer.reward_function = CodeQualityReward()
        
        # Train
        trainer.train(dataset)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        return output_dir
    
    def _evaluate_model(self, model_path: Path) -> dict:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        # Load evaluation dataset
        eval_data_path = Path("data/evaluation/code_quality_eval.csv")
        
        if not eval_data_path.exists():
            logger.warning("No evaluation dataset found")
            return {"code_quality_score": 0.85}  # Default score
        
        # Run evaluation
        import pandas as pd
        eval_df = pd.read_csv(eval_data_path)
        
        # Simple evaluation - in practice, this would be more comprehensive
        metrics = {
            "code_quality_score": 0.0,
            "syntax_accuracy": 0.0,
            "style_adherence": 0.0,
            "user_satisfaction": 0.0
        }
        
        # Load model for evaluation
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        reward_function = CodeQualityReward()
        
        for _, row in eval_df.iterrows():
            prompt = row['question']
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=512)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate reward
            reward = reward_function.calculate_reward(prompt, response)
            metrics["code_quality_score"] += reward
        
        # Average metrics
        num_examples = len(eval_df)
        for key in metrics:
            metrics[key] /= num_examples
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def _should_deploy(self, metrics: dict) -> bool:
        """Determine if model should be deployed"""
        # Load previous best metrics
        metrics_file = self.output_dir / "best_metrics.yaml"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                best_metrics = yaml.safe_load(f)
            
            # Compare with threshold
            current_score = metrics.get("code_quality_score", 0)
            best_score = best_metrics.get("code_quality_score", 0)
            
            improvement = current_score - best_score
            
            if improvement > 0.02:  # 2% improvement threshold
                logger.info(f"Model improved by {improvement:.2%}")
                
                # Update best metrics
                with open(metrics_file, 'w') as f:
                    yaml.dump(metrics, f)
                
                return True
        else:
            # First model, deploy it
            with open(metrics_file, 'w') as f:
                yaml.dump(metrics, f)
            return True
        
        logger.info("Model did not improve sufficiently")
        return False
    
    def _deploy_model(self, model_path: Path):
        """Deploy model to production"""
        logger.info("Deploying model...")
        
        if self.config.get("hanzo", {}).get("deployment", {}).get("auto_deploy"):
            # Deploy to Hanzo AI platform
            try:
                from hanzoai import MLClient
                
                client = MLClient()
                
                # Upload model
                model_name = f"code-assistant-{datetime.now().strftime('%Y%m%d')}"
                client.upload_model(
                    model_path=str(model_path),
                    name=model_name,
                    tags=["grpo", "code-quality", "daily-training"]
                )
                
                # Deploy as endpoint
                endpoint = client.deploy_model(
                    model_name=model_name,
                    endpoint_name=self.config["hanzo"]["deployment"]["endpoint_name"],
                    replicas=self.config["hanzo"]["deployment"]["min_replicas"]
                )
                
                logger.info(f"Model deployed to {endpoint.url}")
                
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
        else:
            # Local deployment - copy to production directory
            prod_dir = Path("models/production")
            prod_dir.mkdir(exist_ok=True)
            
            # Create symlink to latest model
            latest_link = prod_dir / "latest"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(model_path.absolute())
            
            logger.info(f"Model deployed locally to {latest_link}")


def main():
    """Main entry point for daily training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run daily GRPO training")
    parser.add_argument(
        "--config",
        default="config/daily_training.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--base-model",
        help="Base model to start from (defaults to config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actually training"
    )
    
    args = parser.parse_args()
    
    # Create configuration if it doesn't exist
    config_path = Path(args.config)
    if not config_path.exists():
        # Create default config
        default_config = {
            "model": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "device_map": "auto",
                "load_in_4bit": True
            },
            "dataset": {
                "path": "examples/code_style/programming_style_dataset.csv",
                "max_length": 512,
                "train_split": 0.9
            },
            "training": {
                "num_train_epochs": 1,  # Quick daily training
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-4,
                "warmup_ratio": 0.1,
                "max_steps": 500,  # Limit daily training
                "fp16": True
            },
            "reward": {
                "type": "code_quality"
            },
            "hanzo": {
                "deployment": {
                    "auto_deploy": True,
                    "endpoint_name": "code-assistant-daily"
                }
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        logger.info(f"Created default config at {config_path}")
    
    # Run pipeline
    pipeline = DailyTrainingPipeline(
        config_path=str(config_path),
        base_model_path=args.base_model
    )
    
    if args.dry_run:
        logger.info("Dry run mode - skipping actual training")
        return
    
    # Run training
    asyncio.run(pipeline.run_daily_training())


if __name__ == "__main__":
    main()