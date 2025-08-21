#!/bin/bash
# EC2 Setup Script for Evo-2 Models
# Usage: chmod +x setup_ec2_evo2.sh && ./setup_ec2_evo2.sh

set -e  # Exit on error

echo "🚀 Setting up EC2 instance for RNA-Prophet with Evo-2 support"
echo "================================================================"

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ] || [ "$(head -c 3 /sys/hypervisor/uuid 2>/dev/null)" != "ec2" ]; then
    echo "⚠️  This script is designed for EC2 instances"
    echo "   For local setup, use the requirements.txt instead"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
echo "🔧 Installing basic dependencies..."
sudo apt install -y \
    python3.9 \
    python3.9-pip \
    python3.9-venv \
    python3.9-dev \
    git \
    htop \
    nvtop \
    tmux \
    curl \
    wget \
    unzip

# Check for NVIDIA GPU and install drivers
echo "🎮 Checking for GPU..."
if lspci | grep -i nvidia > /dev/null; then
    echo "✅ NVIDIA GPU detected"
    
    # Install NVIDIA drivers if not present
    if ! command -v nvidia-smi &> /dev/null; then
        echo "📥 Installing NVIDIA drivers..."
        sudo apt install -y nvidia-driver-525
        echo "🔄 Reboot required after driver installation"
        echo "   Run this script again after reboot"
        exit 0
    else
        echo "✅ NVIDIA drivers already installed"
        nvidia-smi
    fi
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU mode"
fi

# Create project directory
echo "📁 Setting up RNA-Prophet..."
cd /home/ubuntu  # Assuming default EC2 user

# Clone or update RNA-Prophet
if [ -d "rna-prophet" ]; then
    echo "📥 Updating existing RNA-Prophet..."
    cd rna-prophet
    git pull
else
    echo "📥 Cloning RNA-Prophet..."
    git clone https://github.com/xdlikr/rna-prophet.git
    cd rna-prophet
fi

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (if GPU available)
echo "🔥 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   Installing with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   Installing CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install RNA-Prophet dependencies
echo "🧬 Installing RNA-Prophet dependencies..."
pip install -r requirements.txt

# Install additional dependencies for large models
echo "⚡ Installing optimization libraries..."
pip install accelerate bitsandbytes optimum

# Install ViennaRNA
echo "🔬 Installing ViennaRNA..."
sudo apt install -y viennarna-dev
# Also install via conda if available
if command -v conda &> /dev/null; then
    conda install -c bioconda viennarna -y
fi

# Test basic setup
echo "🧪 Testing basic setup..."
python test_minimal.py

# Create useful scripts
echo "📝 Creating helper scripts..."

# Start script
cat > start_rna_prophet.sh << 'EOF'
#!/bin/bash
# Start RNA-Prophet environment
cd /home/ubuntu/rna-prophet
source venv/bin/activate
echo "🧬 RNA-Prophet environment activated"
echo "💡 Try: python main.py info --models"
echo "🧪 Test models: python test_models.py"
exec bash
EOF
chmod +x start_rna_prophet.sh

# Monitor script
cat > monitor.py << 'EOF'
#!/usr/bin/env python3
"""Monitor system resources during RNA-Prophet training."""
import psutil
import GPUtil
import time
import sys

def monitor_resources():
    """Monitor CPU, memory, and GPU usage."""
    print("🖥️  System Monitoring (Ctrl+C to stop)")
    print("=" * 50)
    
    try:
        while True:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% ({memory.used//1e9:.1f}GB/{memory.total//1e9:.1f}GB)")
            
            # GPU (if available)
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    print(f"GPU{i}: {gpu.load*100:5.1f}% | VRAM: {gpu.memoryUtil*100:5.1f}% ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)")
            except:
                pass
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_resources()
EOF
chmod +x monitor.py

# Training script template
cat > train_example.sh << 'EOF'
#!/bin/bash
# Example training script for Evo-2
source venv/bin/activate

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Example training commands
echo "🧬 RNA-Prophet Training Examples"
echo "Choose a model based on your GPU memory:"
echo

echo "1. DNABERT-2 (4GB+ GPU):"
echo "   python main.py train data/your_data.csv --embedding-model dnabert2"
echo

echo "2. DNABERT-2 Large (8GB+ GPU):"
echo "   python main.py train data/your_data.csv --embedding-model dnabert2_large"
echo

echo "3. Evo (12GB+ GPU):"
echo "   python main.py train data/your_data.csv --embedding-model evo --n-components 256"
echo

echo "4. Evo-2 (24GB+ GPU - when available):"
echo "   python main.py train data/your_data.csv --embedding-model evo2 --n-components 512"
echo

echo "💡 Monitor resources with: python monitor.py"
echo "🧪 Test models first with: python test_models.py"
EOF
chmod +x train_example.sh

# Set up automatic activation
echo "🔧 Setting up automatic environment activation..."
echo 'cd /home/ubuntu/rna-prophet && source venv/bin/activate' >> ~/.bashrc

# Create systemd service for long-running tasks
sudo tee /etc/systemd/system/rna-prophet.service > /dev/null << EOF
[Unit]
Description=RNA-Prophet Training Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rna-prophet
Environment=PATH=/home/ubuntu/rna-prophet/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/rna-prophet/venv/bin/python main.py train /home/ubuntu/data.csv
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "🎉 EC2 setup complete!"
echo
echo "📋 Next steps:"
echo "1. Upload your data to /home/ubuntu/rna-prophet/data/"
echo "2. Test models: python test_models.py"
echo "3. Start training: ./train_example.sh"
echo "4. Monitor resources: python monitor.py"
echo
echo "🔧 Useful commands:"
echo "   source start_rna_prophet.sh  # Activate environment"
echo "   python main.py info --models # Check available models"
echo "   htop                         # Monitor CPU"
echo "   nvidia-smi                   # Monitor GPU"
echo
echo "⚡ For long training jobs, use tmux:"
echo "   tmux new-session -d -s training"
echo "   tmux send-keys 'python main.py train data.csv' Enter"
echo "   tmux attach -t training"
echo
echo "Happy RNA modeling! 🧬"