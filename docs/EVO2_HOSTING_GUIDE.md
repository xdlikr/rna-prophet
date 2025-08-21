# 🚀 Evo-2 Hosting & Deployment Guide

This guide covers different approaches for hosting and using Evo-2 models with RNA-Prophet.

## Current Evo-2 Status

**✅ Update**: Evo-2 is now publicly available! The official models from Arc Institute are:
- `arcinstitute/evo2_7b` - 7B parameters, 1M context (recommended for most use cases)
- `arcinstitute/evo2_40b` - 40B parameters, 1M context (requires multiple GPUs)
- `arcinstitute/evo2_7b_base` - 7B parameters, 8K context
- `arcinstitute/evo2_40b_base` - 40B parameters, 8K context
- `arcinstitute/evo2_1b_base` - 1B parameters, 8K context

**Requirements**: NVIDIA GPU with Compute Capability 8.9+ (Ada/Hopper/Blackwell), CUDA 12.1+, Python 3.12

## Option 1: AWS GPU EC2 (Recommended for Production)

### Instance Recommendations

| Model | Instance Type | GPU | Memory | Cost/hour* |
|-------|---------------|-----|---------|------------|
| `evo2_7b` | `g6.2xlarge` | L4 24GB | 32GB | ~$0.90 |
| `evo2_7b` | `g5.4xlarge` | A10G 24GB | 64GB | ~$1.60 |
| `evo2_40b` | `p4d.24xlarge` | 8x A100 40GB | 1.1TB | ~$30+ |
| `evo2_40b` | `g5.48xlarge` | 8x A10G 24GB | 768GB | ~$16.30 |

*Prices vary by region and change frequently
**Note**: Requires Ada/Hopper/Blackwell GPUs (Compute Capability 8.9+)

### Setup Script for EC2

```bash
#!/bin/bash
# EC2 setup script for Evo-2

# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (for GPU instances) - CUDA 12.1+ required
sudo apt install -y nvidia-driver-535
sudo reboot  # Reboot required

# Install Python 3.12 (required for Evo-2)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.12 python3.12-pip python3.12-venv python3.12-dev git

# Clone RNA-Prophet
git clone https://github.com/xdlikr/rna-prophet.git
cd rna-prophet

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate

# Install CUDA toolkit and Evo-2 prerequisites
pip install --upgrade pip
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation

# Install Evo-2 
pip install evo2

# Install RNA-Prophet dependencies
pip install -r requirements.txt

# Test setup
python test_models.py
```

### EC2 Launch Template

```yaml
# CloudFormation template for Evo-2 instance
Resources:
  Evo2Instance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: g5.4xlarge
      ImageId: ami-0c7217cdde317cfec  # Deep Learning AMI (Ubuntu)
      SecurityGroupIds: 
        - !Ref Evo2SecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          cd /home/ubuntu
          git clone https://github.com/xdlikr/rna-prophet.git
          cd rna-prophet
          # Run setup script
```

## Option 2: Hugging Face Inference Endpoints

### Setup Inference Endpoint

```python
# deploy_evo2_endpoint.py
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    "evo2-rna-prophet",
    repository="arcinstitute/evo2_7b",  # Official Evo-2 model
    framework="pytorch",
    accelerator="gpu",
    instance_size="x4",  # 4x A10G GPUs
    instance_type="nvidia-a10g",
    region="us-east-1",
    vendor="aws",
    account_id="your-hf-account",
    min_replica=0,
    max_replica=1,
    scale_to_zero_timeout=900,
)

print(f"Endpoint URL: {endpoint.url}")
```

### Custom RNA-Prophet Integration

```python
# src/features/embeddings_hf_endpoint.py
import requests
import numpy as np
from typing import List

class HuggingFaceEmbedder:
    """Use Hugging Face Inference Endpoint for embeddings."""
    
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def extract_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings via API."""
        response = requests.post(
            self.endpoint_url,
            headers=self.headers,
            json={
                "inputs": sequences,
                "options": {"wait_for_model": True}
            }
        )
        
        if response.status_code == 200:
            return np.array(response.json())
        else:
            raise RuntimeError(f"API Error: {response.status_code}")

# Usage in RNA-Prophet
embedder = HuggingFaceEmbedder(
    endpoint_url="https://api-inference.huggingface.co/models/evo2-endpoint",
    api_key="hf_your_api_key"
)
```

## Option 3: Google Colab Pro (For Experimentation)

### Colab Setup Notebook

```python
# RNA-Prophet Evo-2 Colab Setup
!pip install -q transformers torch accelerate
!git clone https://github.com/xdlikr/rna-prophet.git
%cd rna-prophet

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Test Evo-2 (if available)
!python test_models.py
```

**Colab Pro Requirements:**
- Colab Pro+ recommended for Evo-2 large
- A100 GPU access needed
- Session timeouts limit long training runs

## Option 4: Local High-End Workstation

### Hardware Requirements

**For `evo2` (131k context):**
- GPU: RTX 4090 (24GB) or A6000 (48GB)
- RAM: 64GB+ system memory
- Storage: 1TB+ NVMe SSD
- Estimated cost: $8,000-15,000

**For `evo2_large` (650B params):**
- GPU: 4x RTX 4090 or 2x A100 80GB
- RAM: 256GB+ system memory  
- Storage: 4TB+ NVMe SSD
- Estimated cost: $25,000-50,000

## Option 5: NVIDIA NIM API (Recommended for API Access)

### NVIDIA Hosted API Integration

```python
# src/features/embeddings_nvidia.py
import requests
from typing import List
import numpy as np

class NVIDIAEvo2Embedder:
    """Use NVIDIA hosted API for Evo-2 embeddings."""
    
    def __init__(self, api_key: str, model: str = "evo2-40b"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://health.api.nvidia.com/v1/biology/arc"
    
    def extract_embeddings(self, sequences: List[str]) -> np.ndarray:
        """Extract embeddings via NVIDIA API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        embeddings = []
        for sequence in sequences:
            payload = {
                "sequence": sequence,
                "return_embeddings": True,
                "layer_name": "blocks.28.mlp.l3"  # Intermediate layer recommended
            }
            
            response = requests.post(
                f"{self.base_url}/{self.model}/generate",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings.append(data["embeddings"])
            else:
                raise RuntimeError(f"API Error: {response.text}")
        
        return np.array(embeddings)
```

## Recommended Approach by Use Case

### 🔬 Research & Development
```bash
# Start with available models
python main.py train data.csv --embedding-model evo --n-components 256

# Now available: Use Evo-2 7B for experiments
python main.py train data.csv --embedding-model evo2_7b --n-components 256
```

### 🏭 Production Pipeline (Small Scale)
```bash
# AWS EC2 g6.2xlarge with Evo-2 7B (most cost-effective)
python main.py train data.csv --embedding-model evo2_7b
```

### 🏢 Production Pipeline (Large Scale)
```bash
# NVIDIA NIM API or HuggingFace Endpoints
# Auto-scaling, managed infrastructure
export NVIDIA_API_KEY="your_key"
python main.py train data.csv --embedding-model evo2_api
```

### 💰 Cost-Conscious
```bash
# Use DNABERT-2 large (still excellent for most cases)
python main.py train data.csv --embedding-model dnabert2_large
```

## Performance Comparison

| Deployment | Setup Time | Cost/1000 seqs* | Scalability | Maintenance |
|------------|------------|------------------|-------------|-------------|
| Local GPU | Days | $0 | Limited | High |
| EC2 On-demand | Hours | $5-50 | Good | Medium |
| EC2 Spot | Hours | $1-15 | Good | Medium |
| HF Endpoints | Minutes | $10-30 | Excellent | Low |
| Together API | Minutes | $20-40 | Excellent | None |
| Colab Pro | Minutes | $3-10 | Poor | None |

*Estimates based on model size and processing time

## Implementation in RNA-Prophet

### Add API Support

```python
# config/embedding_config.yaml
models:
  evo2_api:
    name: "together://togethercomputer/evo-1-131k-base"
    max_length: 131072
    batch_size: 4
    api_key_env: "TOGETHER_API_KEY"
    description: "Evo-2 via Together AI API"

  evo2_hf_endpoint:
    name: "hf://your-evo2-endpoint"
    max_length: 131072
    batch_size: 8
    api_key_env: "HF_API_KEY"
    description: "Evo-2 via HuggingFace Endpoint"
```

### Usage with API

```bash
# Set API key
export TOGETHER_API_KEY="your_key_here"

# Use Evo-2 via API
python main.py train data.csv --embedding-model evo2_api

# Or use environment variable
TOGETHER_API_KEY=your_key python main.py train data.csv --embedding-model evo2_api
```

## Getting Started Recommendations

### Phase 1: Validate Approach (Now)
```bash
# Use available models to prove RNA-Prophet value
python main.py train data.csv --embedding-model dnabert2_large
```

### Phase 2: Upgrade to Evo-2 (Available Now!)
```bash
# Install Evo-2 with proper setup (Python 3.12, CUDA 12.1+)
pip install evo2
python main.py train data.csv --embedding-model evo2_7b
```

### Phase 3: Scale Production (Ready for Deployment)
```bash
# Choose based on scale and budget:
# - Small scale: EC2 g6.2xlarge with Evo-2 7B
# - Large scale: NVIDIA NIM API
# - Enterprise: Self-hosted Evo-2 40B cluster
```

## Monitoring and Costs

### Cost Tracking Script

```python
# scripts/track_costs.py
import boto3
import datetime

def get_ec2_costs(instance_id, days=7):
    """Track EC2 costs for Evo-2 instance."""
    ce = boto3.client('ce')
    
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start.strftime('%Y-%m-%d'),
            'End': end.strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['BlendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )
    
    return response['ResultsByTime']
```

## Quick Start with Official Evo-2

```bash
# Install prerequisites (Python 3.12 required!)
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation

# Install Evo-2
pip install evo2

# Test installation
python -m evo2.test.test_evo2_generation --model_name evo2_7b

# Use with RNA-Prophet
python main.py train data.csv --embedding-model evo2_7b
```

**Hardware Requirements:**
- GPU: NVIDIA Ada/Hopper/Blackwell (RTX 4090, A10G, A100, H100)
- CUDA: 12.1+ with compatible drivers
- Memory: 24GB+ GPU memory for evo2_7b

The future of RNA modeling is here with official Evo-2 support! 🧬🚀