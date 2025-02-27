import os
import re
import json
import logging
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPImageProcessor
import torch.nn.functional as F


# Base CLIP encoder classes
class CLIPTextEncoder:
    """
    A wrapper class for CLIP text encoding functionality.
    Handles tokenization and encoding of text inputs to generate embeddings.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CLIP text encoder.
        
        Args:
            model_name (str): Pretrained CLIP model name
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = 512  # CLIP base model embedding dimension
        
    def encode_batch(
        self,
        texts: List[str],   
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts (List[str]): List of text strings to encode
            batch_size (int): Number of texts to process at once
            
        Returns:
            torch.Tensor: Normalized text embeddings
        """
        embeddings = []
        
        # Process texts in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text batch"):
            batch = texts[i:i + batch_size]
            
            # Tokenize the batch of texts
            inputs = self.processor(
                batch,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                max_length=77  # CLIP's maximum context length
            ).to(self.device)
            
            # Generate embeddings without computing gradients
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                # Normalize embeddings for cosine similarity
                batch_embeddings = F.normalize(outputs, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu())
                
        # Concatenate all batch embeddings
        return torch.cat(embeddings, dim=0) 

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single text into an embedding.
        
        Args:
            text (str): Text string to encode
            
        Returns:
            torch.Tensor: Normalized text embedding
        """
        # Tokenize the text
        inputs = self.processor(
            text,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            max_length=77
        ).to(self.device)
        
        # Generate embedding without computing gradients
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Normalize embedding for cosine similarity
            return F.normalize(outputs, p=2, dim=1).cpu()


class CLIPImageEncoder:
    """
    A wrapper class for CLIP image encoding functionality.
    Processes images and generates embeddings in the CLIP joint embedding space.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CLIP image encoder.
        
        Args:
            model_name (str): Pretrained CLIP model name
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.embedding_dim = 512  # CLIP base model embedding dimension
        
    def encode_batch(self, images: List[Image.Image], batch_size: int = 1024) -> torch.Tensor:
        """
        Encode a batch of images into embeddings.
        
        Args:
            images (List[Image.Image]): List of PIL images to encode
            batch_size (int): Number of images to process at once
            
        Returns:
            torch.Tensor: Normalized image embeddings
        """
        embeddings = []
        
        # Process images in batches
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i + batch_size]
            
            # Preprocess the batch of images
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            
            # Generate embeddings without computing gradients
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Normalize embeddings for cosine similarity
                batch_embeddings = F.normalize(outputs, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all batch embeddings
        return torch.cat(embeddings, dim=0)
        
    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a single image into an embedding.
        
        Args:
            image (Image.Image): PIL image to encode
            
        Returns:
            torch.Tensor: Normalized image embedding
        """
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate embedding without computing gradients
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Normalize embedding for cosine similarity
            return F.normalize(outputs, p=2, dim=1).cpu()


# Base Encoder class to standardize functionality
class BaseEncoder:
    """Base class for all encoders with common functionality"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.batch_size = 32
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing parentheses and handling non-string inputs"""
        if not isinstance(text, str):
            return ''
        return re.sub(r'\(.*?\)', '', text)

    def _save_embeddings(self, embeddings: torch.Tensor, ids: List, file_path: str):
        """Save embeddings and ids to an H5 file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings, compression='gzip')
            f.create_dataset('ids', data=ids)
        self.logger.info(f"Saved embeddings to {file_path}")
        print(f"Saved embeddings to {file_path}")


# Entity Text Encoder
class EntityTextEncoder(BaseEncoder):
    """Encoder for entity text data"""
    
    def __init__(self, embedding_path: str, 
                 qid2label_file: str = None, 
                 pid2label_file: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.encoder = CLIPTextEncoder(device=device)
        self.embedding_file = embedding_path
        
        # Default paths if not provided
        self.qid2label_file = qid2label_file or "/workspace/KGMEL/data/KB/QID2Label.tsv"
        self.pid2label_file = pid2label_file or "/workspace/KGMEL/data/KB/PID2Label.tsv"
    
    def process(self, batch_size: int = 1024):
        """Process entity text data and generate embeddings"""
        print(f"Creating entity text embeddings: {self.embedding_file}")
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        
        # Load QID and PID data
        qid_df = pd.read_csv(self.qid2label_file, sep='\t', names=['id', 'label', 'description'])
        pid_df = pd.read_csv(self.pid2label_file, sep='\t', names=['id', 'label', 'description'])
        
        # Combine entity data
        df = pd.concat([qid_df, pid_df])
        print(f"Total entities: {len(df)} - QIDs: {len(qid_df)} - PIDs: {len(pid_df)}")
        
        self._get_embeddings(df, batch_size)
        
    def _get_embeddings(self, df: pd.DataFrame, batch_size: int):
        """Update embeddings for entity text data"""
        ids = df['id'].tolist()
        print(f"Getting embeddings for {len(df)} entities == {len(ids)} ids")
        
        # Format text with label and description
        texts = [f"{label}: {description}" for label, description in 
                zip(df['label'], df['description'].apply(self.clean_text))]
        
        # Generate embeddings
        embeddings = self.encoder.encode_batch(texts, batch_size)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        self._save_embeddings(embeddings, ids, self.embedding_file)
        return embeddings


# Entity Image Encoder
class EntityImageEncoder(BaseEncoder):
    """Encoder for entity image data"""
    
    def __init__(self, embedding_path: str, img_dir: str, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.encoder = CLIPImageEncoder(device=device)
        self.embedding_file = embedding_path
        self.image_dir = img_dir
    
    def process(self, batch_size: int = 1024):
        """Process entity image data and generate embeddings"""
        print(f"Creating entity image embeddings: {self.embedding_file}")
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        
        # Get image files (assuming format QID_0.jpg)
        image_files = list(Path(self.image_dir).glob('*_0.jpg'))
        print(f"Total entity images: {len(image_files)}")
        
        self._get_embeddings(set(image_files), batch_size)
        
    def _get_embeddings(self, image_paths: Set[Path], batch_size: int):
        """Update embeddings for entity image data"""
        images = []
        image_ids = []
        
        # Load images
        for img_path in tqdm(image_paths, desc="Loading entity images"):
            try:
                image_ids.append(img_path.stem)  # Use filename as ID
                images.append(Image.open(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        # Generate embeddings
        embeddings = self.encoder.encode_batch(images, batch_size)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        self._save_embeddings(embeddings, image_ids, self.embedding_file)
        return embeddings


# Mention Text Encoder
class MentionTextEncoder(BaseEncoder):
    """Encoder for mention text data"""
    
    def __init__(self, embedding_path: str, mapping_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.encoder = CLIPTextEncoder(device=device)
        self.embedding_file = embedding_path
        self.mapping_file = mapping_path
        
    def process(self, json_data: Dict, batch_size: int = 1024):
        """Process mention text data and generate embeddings"""
        print(f"Creating mention text embeddings: {self.embedding_file}")
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)
        
        # Load and process text data
        data, mapping = self._load_text_data(json_data)
        print(f"Number of mention texts to encode: {len(data)}")
        
        # Generate embeddings
        embeddings = self.encoder.encode_batch(data, batch_size)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save embeddings and mapping
        self._save_embeddings(embeddings, range(len(data)), self.embedding_file)
        
        # Save mapping separately
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        print(f"Mapping file saved at {self.mapping_file}")
        
        return embeddings, mapping
        
    def _load_text_data(self, data: Dict):
        """Load and process mention text data efficiently"""
        # Use sets for unique entries
        head_entities = set()
        relation_entities = set()
        tail_entities = set()
        sentences = set()
        
        # Process entities from data
        for item in tqdm(data, desc="Processing mention entities"):
            entity_info = item.get('desc', {})
            triples = item.get('triple', {})
            
            for mention in item['mention']:
                desc = entity_info.get(mention, '')
                desc = '' if desc is None else desc
                head_entities.add(f"{mention}: {desc}".strip())
                sentences.add(item['sentence'].strip())
                
                for triple in triples.get(mention, []):
                    relation_entities.add(triple[1].strip())
                    tail_entities.add(triple[2].strip())
        
        # Convert sets to lists
        head_list = list(head_entities)
        relation_list = list(relation_entities)
        tail_list = list(tail_entities)
        sentence_list = list(sentences)
        
        # All data combined
        all_data = head_list + relation_list + tail_list + sentence_list
        
        # Calculate offsets
        head_offset = 0
        relation_offset = len(head_list)
        tail_offset = relation_offset + len(relation_list)
        sentence_offset = tail_offset + len(tail_list)
        
        # Create mappings
        mapping = {
            "head": {head: idx for idx, head in enumerate(head_list, start=head_offset)},
            "relation": {rel: idx for idx, rel in enumerate(relation_list, start=relation_offset)},
            "tail": {tail: idx for idx, tail in enumerate(tail_list, start=tail_offset)},
            "sentence": {sent: idx for idx, sent in enumerate(sentence_list, start=sentence_offset)}
        }
        
        # Log statistics
        print(f"\nMention Entity Statistics:")
        print(f"Head entities: {len(head_list)}")
        print(f"Relation entities: {len(relation_list)}")
        print(f"Tail entities: {len(tail_list)}")
        print(f"Sentence entities: {len(sentence_list)}")
        
        return all_data, mapping
    
    def get_mappings_directly(self, json_data: Dict):
        """Alternative to creating mapping file - return mappings directly for use in memory"""
        _, mapping = self._load_text_data(json_data)
        return mapping


# Mention Image Encoder
class MentionImageEncoder(BaseEncoder):
    """Encoder for mention image data"""
    
    def __init__(self, embedding_path: str, img_dir: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.encoder = CLIPImageEncoder(device=device)
        self.embedding_file = embedding_path
        self.image_dir = img_dir
    
    def process(self, json_data: Dict, batch_size: int = 1024 ):
        """Process mention image data and generate embeddings"""
        print(f"Creating mention image embeddings: {self.embedding_file}")
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)

        image_paths = []
        image_ids = []
        
        for item in json_data:
            image_ids.append(item['id'])
            image_paths.append(self._get_image_path(item['imgPath']))
        
        print(f"Number of mention images to encode from JSON: {len(image_paths)}")
        
        # Load and encode images
        self._get_embeddings(image_paths, image_ids, batch_size)
    
    def _process_from_directory(self, batch_size: int):
        """Process all images in the directory"""
        # Get all JPG images
        image_files = list(Path(self.image_dir).glob('*.jpg'))
        image_ids = [img_path.stem for img_path in image_files]
        
        print(f"Number of mention images to encode from directory: {len(image_files)}")
        
        # Process images
        self._get_embeddings(image_files, image_ids, batch_size)
    
    def _get_image_path(self, imgpath) -> Path:
        """Get full image path"""
        return Path(os.path.join(self.image_dir, f"{imgpath.split('.')[0]}.jpg"))
    
    def _get_embeddings(self, image_paths: List[Path], image_ids: List[str], batch_size: int):
        """Update embeddings for mention image data"""
        processed_images = []
        processed_ids = []
        
        # Load images
        for img_path, img_id in tqdm(zip(image_paths, image_ids), desc="Loading mention images", total=len(image_paths)):
            try:
                processed_images.append(Image.open(img_path))
                processed_ids.append(img_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        print(f"Number of mention images loaded: {len(processed_images)}")
        
        if not processed_images:
            print("No images could be loaded. Check image paths.")
            return None
        
        # Generate embeddings
        embeddings = self.encoder.encode_batch(processed_images, batch_size)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save embeddings
        self._save_embeddings(embeddings, processed_ids, self.embedding_file)
        return embeddings

