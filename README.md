# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking

## Setup 

### 1. Data Preparation

You can download the required datasets from the [MIMIC repository](https://github.com/pengfei-luo/MIMIC).

Extract the image archives to their respective directories:

```bash
# Extract WikiMEL dataset
tar -xzf /data/dataset/image/WikiMEL.tar.gz -C /data/dataset/image/

# Extract WikiDiverse dataset
tar -xzf /data/dataset/image/WikiDiverse.tar.gz -C /data/dataset/image/

# Extract RichpediaMEL dataset
tar -xzf /data/dataset/image/RichpediaMEL.tar.gz -C /data/dataset/image/

# Extract Knowledge Base images
tar -xzf data/KB_image.tar.gz -C /data/
```

After extraction, you should have the following directory structure:
```
/data/
├── dataset/
│   └── image/
│       ├── WikiMEL/
│       ├── WikiDiverse/
│       └── RichpediaMEL/
└── KB/
    └── image/
```

### 2. Create Required Directories

Ensure all required directories exist (most will be created automatically when extracting the data files):

```bash
# The following directories should already exist after data extraction,
# but you can create them if needed
mkdir -p checkpoints
mkdir -p embedding/combined
mkdir -p embedding/entity
mkdir -p embedding/mention
mkdir -p output
```

These directories store various components of the pipeline:
- `checkpoints/`: Saved model checkpoints
- `embedding/`: Precomputed embeddings for entities and mentions
- `output/`: Results and evaluation outputs

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set API Credentials

Set up the necessary API keys and tokens:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key"

# Set Hugging Face token
export HUGGINGFACE_TOKEN="your_huggingface_token"

# Set Hugging Face cache directory (optional)
export HUGGINGFACE_HUB_CACHE="/path/to/huggingface/cache"

# Login to Hugging Face (alternative method)
# huggingface-cli login

# Login to Weights & Biases (wandb)
wandb login

# Alternatively, you can set the WANDB_API_KEY environment variable
# export WANDB_API_KEY="your_wandb_api_key"
```

You can add these to your `.bashrc` or `.bash_profile` for persistence, or include them in your run script for convenience.

### 4. Run the Code

Execute the run script:

```bash
./run.sh
```

## Structure

```
KGMEL/
├── checkpoints/         # Directory for saving trained models
├── data/
│   ├── dataset/
│   │   ├── image/       # Contains extracted image datasets
│   │   ├── mapping/
│   │   ├── RichpediaMEL.json
│   │   ├── WikiDiverse.json
│   │   └── WikiMEL.json
│   └── KB/              # Knowledge Base data with entity triples and mappings
│       ├── image/
│       ├── PID2Label.tsv
│       ├── QID2Label.tsv
│       ├── Triples-RichpediaMEL.tsv
│       ├── Triples-WikiDiverse.tsv
│       └── Triples-WikiMEL.tsv
├── embedding/           # Directory for pre-computed embeddings
├── module/
│   ├── __init__.py
│   ├── generate.py
│   ├── rerank.py        # Entity reranking module
│   └── retrieve.py      # Entity retrieval module
├── output/              # Output directory for results
├── utils/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── embedding_processor.py
│   ├── encoder.py
│   ├── evaluate.py
│   ├── train.py
│   ├── triple_filtering.py
│   └── triple_parser.py
├── main.py              # Main code entry point
├── requirements.txt     # Dpendencies
└── run.sh               # Execution script
```