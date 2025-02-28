# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking

## Setup 

### 0. Data Preparation

Our dataset is based on the [MIMIC repository](https://github.com/pengfei-luo/MIMIC). 

You need to download the following files from our [Dropbox](https://www.dropbox.com/scl/fo/x61o4peh9ji1pww0zobao/AHiBAJq6taVF2-j7bQCKf7E?rlkey=x4pe5akrqaz6benuj376t7rqm&st=pky8etmp&dl=0):
- `triple.tar` - Knowledge graph triples 
- `mention_image.tar` - Mention images 
- `kb_image.tar` - Knowledge base entity images
- 
Place the files in the following locations:
- `mention_image.tar` → Place in the `data/dataset/` directory
- `kb_image.tar` and `triple.tar` → Place in the `data/KB/` directory

Then extract the files:
```bash
# Extract mention images for datasets
tar -xf data/dataset/mention_image.tar -C data/dataset/

# Extract Knowledge Base images
tar -xf data/KB/kb_image.tar -C data/KB/

# Extract Knowledge Graph triples
tar -xf data/KB/triple.tar -C data/KB/
```

After extraction, you should have the following directory structure:
```
├── data/
│   ├── dataset/        # Contains the 3 MEL datasets (WikiMEL, WikiDiverse, RichpediaMEL)
│   │   ├── image/      # Contains extracted image datasets from mention_image.tar
│   │   ├── mapping/    # Contains mapping files for dataset processing
│   │   ├── RichpediaMEL.json
│   │   ├── WikiDiverse.json
│   │   └── WikiMEL.json
│   └── KB/              # Contains Knowledge Base data with including KG triples
│       ├── image/       # Contains extracted image datasets from kb_image.tar
│       ├── PID2Label.tsv
│       ├── QID2Label.tsv
│       ├── Triples-RichpediaMEL.tsv  # These triple files come from triples.tar.gz
│       ├── Triples-WikiDiverse.tsv
│       └── Triples-WikiMEL.tsv
```

### 1. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Set Keys and Tokens
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

### 3. Run the Code
Execute the run script:
```bash
./run.sh
```

## Structure
```
KGMEL/
├── checkpoints/         # Directory for saving trained models
├── data/
│   ├── dataset/        # Contains the 3 MEL datasets (WikiMEL, WikiDiverse, RichpediaMEL)
│   │   ├── image/      # Contains extracted image datasets from mention_image.tar
│   │   ├── mapping/    # Contains mapping files for dataset processing
│   │   ├── RichpediaMEL.json
│   │   ├── WikiDiverse.json
│   │   └── WikiMEL.json
│   └── KB/              # Contains Knowledge Base data with including KG triples
│       ├── image/       # Contains extracted image datasets from kb_image.tar
│       ├── PID2Label.tsv
│       ├── QID2Label.tsv
│       ├── Triples-RichpediaMEL.tsv  # These triple files come from triples.tar.gz
│       ├── Triples-WikiDiverse.tsv
│       └── Triples-WikiMEL.tsv
├── embedding/           # Directory for pre-computed embeddings
├── module/
│   ├── __init__.py
│   ├── generate.py      # Triple Generation module
│   ├── retrieve.py      # Candidate Entity Retrieval module
│   └── rerank.py        # Entity Reranking module
├── output/              # Output directory for results
├── utils/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── embedding_processor.py
│   ├── encoder.py
│   ├── evaluate.py
│   ├── train.py
│   ├── triple_filtering.py
│   └── triple_parser.py
├── main.py              # Main code 
├── requirements.txt     # Dependencies
└── run.sh               # Execution script
```
