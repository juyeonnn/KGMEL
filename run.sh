#!/bin/bash

# Add current directory to Python path
export PYTHONPATH=${PYTHONPATH}:$(pwd)

# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key"

# Set Hugging Face token
export HUGGINGFACE_TOKEN="your_huggingface_token"

# Set GPU number
export GPU=0

# Run the Python script with the specified GPU
CUDA_VISIBLE_DEVICES=$GPU python main.py