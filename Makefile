.PHONY: help setup train evaluate deploy test clean docker-build docker-up docker-down

# Colors
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)GRPO - Guided Reinforcement Policy Optimization$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Set up development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "$(GREEN)Setup complete!$(NC)"

train: ## Run training with default config
	@echo "$(YELLOW)Starting training...$(NC)"
	python src/train.py --config config/grpo_config.yaml

train-hanzo: ## Run training with Hanzo integration
	@echo "$(YELLOW)Starting training with Hanzo AI...$(NC)"
	python src/train.py --config config/grpo_config.yaml --hanzo

collect: ## Collect code examples from Hanzo Chat
	@echo "$(YELLOW)Collecting code examples...$(NC)"
	python scripts/collect_code_examples.py

daily-train: ## Run daily training pipeline
	@echo "$(YELLOW)Running daily training pipeline...$(NC)"
	python scripts/daily_training.py

evaluate: ## Evaluate trained model
	@echo "$(YELLOW)Evaluating model...$(NC)"
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: MODEL parameter required. Usage: make evaluate MODEL=path/to/model$(NC)"; \
		exit 1; \
	fi
	python scripts/evaluate_model.py --model-path $(MODEL)

cli: ## Run CLI tool
	@echo "$(BLUE)GRPO CLI$(NC)"
	python -m grpo.cli

generate: ## Generate text with trained model
	@if [ -z "$(MODEL)" ] || [ -z "$(PROMPT)" ]; then \
		echo "$(RED)Error: MODEL and PROMPT required. Usage: make generate MODEL=path PROMPT='text'$(NC)"; \
		exit 1; \
	fi
	python -m grpo.cli generate $(MODEL) --prompt "$(PROMPT)"

test: ## Run tests
	@echo "$(YELLOW)Running tests...$(NC)"
	pytest tests/ -v --cov=src/grpo --cov-report=term-missing

lint: ## Run linting
	@echo "$(YELLOW)Running linters...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	mypy src/

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Clean complete!$(NC)"

# Docker commands
docker-build: ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(NC)"
	docker-compose build

docker-up: ## Start Docker services
	@echo "$(YELLOW)Starting Docker services...$(NC)"
	docker-compose up -d

docker-down: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	docker-compose down

docker-train: ## Run training in Docker
	@echo "$(YELLOW)Running training in Docker...$(NC)"
	docker-compose run --rm grpo-training

docker-logs: ## View Docker logs
	docker-compose logs -f

# Hanzo AI commands
hanzo-login: ## Login to Hanzo AI
	@echo "$(YELLOW)Logging in to Hanzo AI...$(NC)"
	hanzo auth login

hanzo-deploy: ## Deploy model to Hanzo AI
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: MODEL parameter required. Usage: make hanzo-deploy MODEL=path/to/model$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Deploying to Hanzo AI...$(NC)"
	python scripts/deploy_to_hanzo.py --model $(MODEL)

hanzo-status: ## Check Hanzo AI status
	@echo "$(YELLOW)Checking Hanzo AI status...$(NC)"
	python -m grpo.cli hanzo status

# Development shortcuts
dev: ## Start development environment
	@echo "$(YELLOW)Starting development environment...$(NC)"
	docker-compose up -d grpo-dev
	@echo "$(GREEN)Jupyter available at: http://localhost:8888$(NC)"
	@echo "$(GREEN)TensorBoard available at: http://localhost:6006$(NC)"

tensorboard: ## Start TensorBoard
	@echo "$(YELLOW)Starting TensorBoard...$(NC)"
	tensorboard --logdir outputs/

# Dataset commands
dataset-validate: ## Validate dataset
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)Error: FILE parameter required. Usage: make dataset-validate FILE=path/to/dataset.csv$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Validating dataset...$(NC)"
	python -m grpo.cli dataset validate --input $(FILE)

dataset-create: ## Create dataset template
	@echo "$(YELLOW)Creating dataset template...$(NC)"
	python -m grpo.cli dataset create --output template_dataset.csv

# Installation targets
install-cuda: ## Install CUDA dependencies
	@echo "$(YELLOW)Installing CUDA dependencies...$(NC)"
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

install-cpu: ## Install CPU-only dependencies
	@echo "$(YELLOW)Installing CPU-only dependencies...$(NC)"
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu