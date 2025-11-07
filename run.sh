#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip torch torchvision transformers pillow

# Clone or update OmniGen2
if [ ! -d "OmniGen2" ]; then
    git clone https://github.com/VectorSpaceLab/OmniGen2.git
else
    echo "OmniGen2 already exists, pulling latest..."
    cd OmniGen2 && git pull && cd ..
fi

pip install -r OmniGen2/requirements.txt

python generate_images.py