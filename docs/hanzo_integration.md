# Hanzo AI Platform Integration Guide

This guide explains how to use GRPO with the Hanzo AI cloud platform for scalable model fine-tuning and deployment.

## Prerequisites

1. **Hanzo AI Account**: Sign up at [hanzo.ai](https://hanzo.ai)
2. **Hanzo CLI**: Install the command-line interface
3. **API Credentials**: Obtain from your Hanzo dashboard

## Installation

```bash
# Install Hanzo CLI and ML tools
pip install hanzoai-cli hanzoai-ml

# Configure authentication
hanzo auth login
```

## Quick Start

### 1. Create a New Project

```bash
# Create a GRPO project
hanzo ml create my-grpo-project --type=reinforcement-learning

# List projects
hanzo ml list-projects
```

### 2. Upload Your Dataset

```bash
# Upload local dataset
hanzo ml dataset upload \
  --file=examples/skippy/skippy_knowledge_base.csv \
  --project=my-grpo-project \
  --name=skippy-dataset

# Or upload from S3
hanzo ml dataset import \
  --source=s3://my-bucket/data.csv \
  --project=my-grpo-project
```

### 3. Configure Training

Create a Hanzo-specific configuration:

```yaml
# hanzo_config.yaml
hanzo:
  project_name: "my-grpo-project"
  experiment_name: "experiment-1"
  
  # Resource allocation
  resources:
    gpu_type: "A100"  # Options: T4, V100, A100, H100
    gpu_count: 1
    memory: "32Gi"
    cpu: "8"
  
  # Auto-scaling
  scaling:
    min_replicas: 1
    max_replicas: 5
    target_gpu_utilization: 80
  
  # Cost optimization
  spot_instances: true
  preemptible: true

# Include your GRPO config
inherit: config/grpo_config.yaml
```

### 4. Start Training

```bash
# Start training job
hanzo ml train \
  --config=hanzo_config.yaml \
  --project=my-grpo-project \
  --watch  # Stream logs

# Check status
hanzo ml status --job-id=<job-id>
```

## Advanced Features

### Distributed Training

```bash
# Multi-GPU training
hanzo ml train \
  --config=config.yaml \
  --distributed \
  --nodes=2 \
  --gpus-per-node=8
```

### Hyperparameter Tuning

```yaml
# hparam_search.yaml
search:
  strategy: "bayesian"  # or "grid", "random"
  metric: "eval_loss"
  mode: "min"
  
  parameters:
    learning_rate:
      type: "float"
      min: 1e-5
      max: 5e-4
    
    lora_r:
      type: "choice"
      values: [8, 16, 32, 64]
    
    batch_size:
      type: "choice"
      values: [1, 2, 4, 8]
  
  trials: 20
  parallel_trials: 4
```

```bash
# Run hyperparameter search
hanzo ml tune \
  --config=hparam_search.yaml \
  --base-config=grpo_config.yaml \
  --project=my-grpo-project
```

### Model Registry

```bash
# Register trained model
hanzo ml register \
  --model-path=outputs/checkpoint-1000 \
  --name=skippy-assistant \
  --version=1.0.0 \
  --tags=production,grpo

# List models
hanzo ml models list --project=my-grpo-project

# Download model
hanzo ml models download \
  --name=skippy-assistant \
  --version=1.0.0 \
  --output=./models
```

### Deployment

```bash
# Deploy model as API endpoint
hanzo ml deploy \
  --model=skippy-assistant:1.0.0 \
  --name=skippy-api \
  --replicas=2 \
  --gpu=T4

# Get endpoint details
hanzo ml endpoints describe --name=skippy-api

# Test endpoint
curl -X POST https://api.hanzo.ai/v1/endpoints/skippy-api/generate \
  -H "Authorization: Bearer $HANZO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I reset a password?",
    "max_tokens": 256
  }'
```

## Monitoring and Observability

### Real-time Metrics

```bash
# View training metrics
hanzo ml logs --job-id=<job-id> --follow

# Export metrics
hanzo ml metrics export \
  --job-id=<job-id> \
  --format=csv \
  --output=metrics.csv
```

### Integration with Monitoring Tools

```python
# Python SDK example
from hanzoai import MLClient
from hanzoai.monitoring import MetricsCollector

client = MLClient()
metrics = MetricsCollector(project="my-grpo-project")

# Custom metrics
metrics.log({
    "custom/reward_score": 0.85,
    "custom/response_quality": 0.92
})

# Get Grafana dashboard URL
dashboard_url = client.get_dashboard_url(project="my-grpo-project")
print(f"Dashboard: {dashboard_url}")
```

## Cost Management

### Resource Optimization

```bash
# Get cost estimate
hanzo ml estimate \
  --config=config.yaml \
  --duration=24h

# View current costs
hanzo billing usage --project=my-grpo-project

# Set cost alerts
hanzo billing alert create \
  --project=my-grpo-project \
  --threshold=1000 \
  --email=alerts@company.com
```

### Spot Instance Management

```yaml
# spot_config.yaml
spot:
  enabled: true
  max_price: 0.50  # USD per hour
  interruption_behavior: "checkpoint"  # or "terminate"
  checkpoint_frequency: 100  # steps
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/train.yml
name: GRPO Training

on:
  push:
    branches: [main]
    paths:
      - 'data/**'
      - 'config/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Hanzo CLI
        run: |
          pip install hanzoai-cli
          hanzo auth login --token=${{ secrets.HANZO_TOKEN }}
      
      - name: Upload Dataset
        run: |
          hanzo ml dataset upload \
            --file=data/training.csv \
            --project=grpo-prod
      
      - name: Start Training
        run: |
          hanzo ml train \
            --config=config/prod.yaml \
            --project=grpo-prod \
            --wait
      
      - name: Deploy if Successful
        if: success()
        run: |
          hanzo ml deploy \
            --latest \
            --project=grpo-prod \
            --name=prod-endpoint
```

## Troubleshooting

### Common Issues

1. **GPU OOM Errors**
   ```bash
   # Reduce batch size or use gradient accumulation
   hanzo ml update-job \
     --job-id=<job-id> \
     --set="training.per_device_train_batch_size=1" \
     --set="training.gradient_accumulation_steps=16"
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision
   hanzo ml update-job \
     --job-id=<job-id> \
     --set="training.fp16=true" \
     --set="training.tf32=true"
   ```

3. **Connection Issues**
   ```bash
   # Check connectivity
   hanzo status
   
   # Use different region
   hanzo config set region us-west-2
   ```

### Debug Mode

```bash
# Enable debug logging
export HANZO_LOG_LEVEL=DEBUG

# Run with verbose output
hanzo ml train --config=config.yaml --verbose --debug

# Interactive debugging
hanzo ml debug --job-id=<job-id> --shell
```

## Best Practices

1. **Version Control**: Always version your datasets and configurations
2. **Checkpointing**: Enable frequent checkpointing for long runs
3. **Monitoring**: Set up alerts for anomalies and failures
4. **Cost Control**: Use spot instances for experimentation
5. **Security**: Rotate API keys regularly and use IAM roles

## Support

- Documentation: [docs.hanzo.ai](https://docs.hanzo.ai)
- Community: [community.hanzo.ai](https://community.hanzo.ai)
- Support: support@hanzo.ai