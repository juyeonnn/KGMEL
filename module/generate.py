"""
GPT Vision-based Image Processing Pipeline
This module processes images using GPT Vision models to generate knowledge graph triples.
"""

import argparse
import json
import os
import base64
from pathlib import Path
import requests
from PIL import Image
from tqdm import tqdm
from typing import Any, Type, Union, List, Dict
import torch
from huggingface_hub import login
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from utils.triple_parser import parse

# -------------------- Main Module --------------------

class TripleGenerationModule:
    def __init__(self, args):
        self.args = args
        if 'gpt' in args.vlm.lower():
            self.processor = GPTVisionProcessor(args.vlm)
        elif 'llava' in args.vlm.lower():
            self.processor = LLaVAProcessor(args.vlm)
        else:
            raise ValueError(f"Invalid VLM model: {args.vlm}")
        
        self.data_processor = DataProcessor(
            dataset_name=args.dataset,
            model_name=args.vlm,
            processor=self.processor,
            base_dir=args.base_dir,
            data_dir = args.data_dir
        )
    def generate(self):
        print('--------- Generating Triples ---------')
        self.data_processor.process_dataset()

# -------------------- Model Processing --------------------

class GPTVisionProcessor:
    """
    Processor for GPT Vision models to generate knowledge graph triples from images and text.
    """
    def __init__(self,model_name: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the GPT Vision processor.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the GPT Vision model to use
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def format_prompt(self, sentence: str, mention: list) -> List[Dict]:
        """
        Create conversation prompt for the model.
        
        Args:
            sentence: Input text
            mention: List of entities to analyze
            
        Returns:
            Formatted conversation for the model
        """
        mention_formatted = [f'"{m}"' for m in set(mention)]
        mention_str = ', '.join(mention_formatted)
        
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert in generating knowledge graphs from images and text, in Wikidata format."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given the image and text {sentence}, please generate triples for the entities {mention_str}. following the steps below:

### Step 1: Entity Type
For each entity in {mention_str}, identify its type, following the format:
- "entity_name": type of entity
Type of entity can be person, nationality, religious group, political group, organization, country, city, state, building, airport, highway, bridge, company, agency, institution, product, event, work of art, law, language, etc.

### Step 2: Entity Description
Provide a description for each entity in {mention_str}, following the format:
- "entity_name": key information 
Focus on factual information that can be inferred from the image and text to describe the entity.

### Step 3: Triples
Finally, use the type and information from steps 1 and 2 to generate knowledge graph triples:
Convert the entity types and information into triples following the format, each triple on a new line:
- "entity_name" | relation1 | entity1
- "entity_name" | relation2 | entity2

Based on the entity type and information provided in the image and text, choose the most relevant relations from the following list to generate triples:
"instance of", "subclass of", "part of", "has characteristic", 
"field of work", "occupation", "sex or gender", "country of citizenship", "position held", "religion or worldview", 
"member of", "owner of", "country", "capital", "continent", "located in", "industry", "participant", "genre", "named after"

ONLY If the image is blank, use your own knowledge to extract the information from the text.
If the image is not blank, focus on the information that can be inferred from the image and text.
PLEASE Extract the MOST relevant triples that can be inferred from the image and text.
Make sure to use the EXACT matching name for each entity ({mention_str}), and follow the format for EACH step. Start with ### Step 1: Entity Type
"""
                    },
                    # Image will be added in the process_image method
                ]
            }
        ]
    
    def process_image(self, image_path: str, sentence: str, mention: list) -> str:
        """
        Process a single image and return the model's response.
        
        Args:
            image_path: Path to the image file
            sentence: Text context for the image
            mention: List of entities to analyze
            
        Returns:
            Model's response with knowledge graph triples
        """
        base64_image = self.encode_image(image_path)
        conversation = self.format_prompt(sentence, mention)
        
        # Add the image to the user message content
        conversation[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
        
        payload = {
            "model": self.model_name,
            "messages": conversation,
            "max_tokens": 1000
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"


class LLaVAProcessor:
    """
    Processor for LLaVA models to generate knowledge graph triples from images and text.
    """
    def __init__(self, model_name: str ):
        """
        Initialize the LLaVA processor.
        
        Args:
            model_name: Name of the LLaVA model to use (e.g., "llava-hf/llava-v1.6-vicuna-13b-hf" or "llava-hf/llava-v1.6-mistral-7b-hf")
        """
        self.model_name = model_name
        login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self._initialize_model()

    def _initialize_model(self):
        """Initialize the LLaVA model and processor."""
        compute_dtype = torch.float16

        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=compute_dtype, 
            low_cpu_mem_usage=True,
            # cache_dir=self.args.huggingface_cache_dir,
        ).to(self.device)
        
        processor = LlavaNextProcessor.from_pretrained(
            self.model_name,
            patch_size=model.config.vision_config.patch_size,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            # cache_dir = self.args.huggingface_cache_dir,
            use_fast=True,
        )

        return model, processor

    def format_prompt(self, sentence: str, mention: list) -> List[Dict]:
        mention_formatted = [f'"{m}"' for m in set(mention)]
        mention_words = ', '.join(mention_formatted)
        
        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are an expert in generating knowledge graphs from images and text."
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given the image and text {sentence}, please generate triples for the entities {mention_words}. following the steps below:

### Step 1: Entity Type
For each entity in {mention_words}, identify its type, following the format:
- "entity_name": type of entity
Type of entity can be person, nationality, religious group,  political group, organization, country, city, state, building, airport, highway, bridge, company, agency, institution,  product, event, work of art, law, language, etc.

### Step 2:  Entity Description
Provide a description for each entity in {mention_words}, following the format:
- "entity_name": key information 
Focus on factual information that can be inferred from the image and text to describe the entity.

### Step 3: Triples
Finally, use the type and information from steps 1 and 2 to generate knowledge graph triples:
Convert the entity types and information into triples following the format, each triple on a new line:
- "entity_name" | relation1 | entity1
- "entity_name" | relation2 | entity2

Based on the entity type and information provided in the image and text, choose the most relevant relations from the following list to generate triples:
"instance of", "subclass of", "part of", "has characteristic", 
"field of work", "occupation", "sex or gender", "country of citizenship", "position held", "religion or worldview", 
"member of", "owner of",  "country",  "capital", "continent", "located in", "industry", "participant", "genre",  "named after"

ONLY If the image is blank,  use your own knowledge to extract the information from the text.
If the image is not blank, focus on the information that can be inferred from the image and text.
PLEASE Extract the MOST relevant triples that can be inferred from the image and text.
Make sure to use the EXACT matching name for each entity ({mention_words}), and follow the format for EACH step. Start with ### Step 1: Entity Type
"""
                    },
                    {"type": "image"}
                ]
            }
        ]
    
    def process_image(self, image_path: str, sentence: str, mention: list) -> str:
        """
        Process a single image and return the model's response.
        
        Args:
            image_path: Path to the image file
            sentence: Text context for the image
            mention: List of entities to analyze
            
        Returns:
            Model's response with knowledge graph triples
        """
        image = Image.open(image_path)
        conversation = self.format_prompt(sentence, mention)
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            repetition_penalty=1.1,
        )
        
        result = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Different models use different delimiters in their outputs
        if "vicuna-13b" in self.model_name:
            # 13B model uses ASSISTANT as delimiter
            return result.split('ASSISTANT')[-1].strip()
        elif "mistral-7b" in self.model_name:
            # 7B model uses [/INST] as delimiter
            return result.split('[/INST]')[-1].strip()


# -------------------- Data Processing --------------------

class DataProcessor:

    def __init__(
    self, 
    dataset_name: str, 
    model_name: str,
    processor: Union[Any, Type['GPTVisionProcessor'], Type['LLaVAProcessor']],
    split: str ='total',
    base_dir: str = '/workspace/KGMEL',
    data_dir: str = '/workspace/KGMEL/data'
    ):
        """
        Initialize the data processor.
        
        Args:
            dataset_name: Name of the dataset to process
            split: Data split (train/val/test)
            gpt_processor: Initialized GPT Vision processor
            dir: Base directory for data files
        """
        self.split              = split
        self.dataset_name       = dataset_name
        self.model_name         = model_name
        self.base_dir           = base_dir
        self.data_dir           = data_dir
        self.processor          = processor
        self.json_file          = os.path.join(self.data_dir, f"dataset/{dataset_name}.json")
        self.mapping_file_path  = os.path.join(self.data_dir, "dataset/mapping/ids_split_mappings.json")
        self.image_dir          = os.path.join(self.data_dir, f"dataset/image/{dataset_name}")
        self.split_id_lst       = self._get_mapping()
        self.output_file        = self._get_output_path()
        self.processed_data     = self._load_existing_data()
        print(f"Loaded {len(self.split_id_lst)} IDs for split {split}")
        print(f"Results will be saved to {self.output_file}")

    def _get_output_path(self) -> str:
        """
        Generate output file path based on the model name.
        
        Returns:
            Path to the output JSON file
        """
            
        base_filename = Path(self.json_file).stem
        model_identifier = self.model_name.split('/')[-1] if '/' in self.model_name else self.processor.model_name

        os.makedirs(os.path.join(self.base_dir, "output"), exist_ok=True)
        return os.path.join(self.base_dir, f"output/{base_filename}_{model_identifier}_{self.split}.json")

    def _get_mapping(self) -> List[str]:
        """Load split mapping from json file."""
        with open(self.mapping_file_path, 'r') as f:
            response = json.load(f)
        if self.split == 'total':
            return response[self.dataset_name]['train'] + response[self.dataset_name]['val'] + response[self.dataset_name]['test']
        else:
            return response[self.dataset_name][self.split]

    def _load_existing_data(self) -> List[Dict]:
        """Load existing processed data or initialize new data."""
        if os.path.exists(self.output_file):
            print(f"Loading existing processed data from {self.output_file}")
            with open(self.output_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Loading data from {self.json_file}")
            with open(self.json_file, 'r') as f:
                return json.load(f)

    def _get_image_path(self, item: Dict) -> str:
        """Get image path based on dataset type."""
        return os.path.join(self.image_dir, f"{item['imgPath'].split('.')[0]}.jpg")

    def process_dataset(self):
        """Process the entire dataset."""
        total_processed = 0

        # Use a unified response key
        self.response_key = "response"

        for idx, item in enumerate(tqdm(self.processed_data)):
            if item['id'] not in self.split_id_lst:
                continue
            if 'response' in item:
                continue
            try: 
                image_path = self._get_image_path(item)
                response = self.processor.process_image(image_path, item['sentence'], item['mention'])

                self.processed_data[idx][self.response_key] = response
                total_processed += 1

                print(f"[ID]: {item.get('id')}")
                print(f"[MENTION]: {item['mention']}")
                print(f"[SENTENCE]: {item['sentence']}")
                print(f"[RESPONSE]: {response}")

                if (total_processed % 50 == 0) or (total_processed == len(self.split_id_lst)):
                    self._save_checkpoint()

            except Exception as e:
                print(f"\nError processing image for ID {item.get('id')}: {str(e)}")
                self.processed_data[idx][self.response_key] = f"Error: {str(e)}"

        self._parse_results()
        self._save_final_results()

    def _save_checkpoint(self):
        """Save checkpoint during processing."""

        processed_count = sum(1 for item in self.processed_data if self.response_key in item)
        total_entries = len(self.processed_data)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved checkpoint after processing {processed_count} images out of {total_entries} entries")

    def _parse_results(self):
        """Parse the processed results."""
        self.processed_data = parse(self.processed_data, key=self.response_key)
    
    def _save_final_results(self):
        """Save final results."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False)


