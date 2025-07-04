#!/usr/bin/env python3
"""
Main training script for GRPO
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from grpo import GRPOTrainer, GRPODataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a model using GRPO")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Override dataset path from config"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--hanzo",
        action="store_true",
        help="Use Hanzo AI platform features"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data:
        config["dataset"]["path"] = args.data
    if args.output:
        config["training"]["output_dir"] = args.output
    
    # Initialize Hanzo integration if requested
    if args.hanzo:
        try:
            from hanzo import HanzoMLClient
            client = HanzoMLClient()
            client.start_experiment(
                project=config["hanzo"]["project_name"],
                name=config["hanzo"]["experiment_name"],
                tags=config["hanzo"]["tags"]
            )
            logger.info("Hanzo ML tracking enabled")
        except ImportError:
            logger.warning("Hanzo ML client not available, continuing without tracking")
    
    # Load dataset
    logger.info(f"Loading dataset from {config['dataset']['path']}")
    dataset = GRPODataset(
        data_path=config["dataset"]["path"],
        max_length=config["dataset"]["max_length"],
        train_split=config["dataset"]["train_split"]
    )
    
    # Initialize trainer
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model_name=config["model"]["name"],
        config=config,
        output_dir=config["training"]["output_dir"]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(dataset)
    
    # Log completion
    if args.hanzo:
        try:
            client.end_experiment()
        except:
            pass
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()