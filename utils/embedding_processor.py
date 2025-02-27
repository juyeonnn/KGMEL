import json
import logging
from pathlib import Path
import torch
import networkx as nx
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd 
import sys
import os
import h5py
import numpy as np
import sys
import os

from utils.encoder import EntityTextEncoder, EntityImageEncoder, MentionTextEncoder, MentionImageEncoder


class EmbeddingProcessor:
    def __init__(self, args):
        self.args = args

        self.device ='cpu' # Fix to CPU to avoid memory issues on GPU

        self.json_path = f"{self.args.base_dir}/output/{self.args.dataset}_{self.args.vlm}_total.json"

        self.entity_text_embedding_path     = f"{self.args.base_dir}/embedding/entity/text_embedding.h5"
        self.entity_img_embedding_path      = f"{self.args.base_dir}/embedding/entity/img_embedding.h5"
        self.mention_text_embedding_path    = f"{self.args.base_dir}/embedding/mention/text_embedding_{self.args.dataset}_{self.args.vlm}.h5" 
        self.mention_text_mapping_path      = f"{self.args.base_dir}/embedding/mention/text_embedding_{self.args.dataset}_{self.args.vlm}_mapping.json"
        self.mention_img_embedding_path     = f"{self.args.base_dir}/embedding/mention/img_embedding_{self.args.dataset}.h5"

        self.combined_node_path             = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_node.h5"
        self.combined_relation_path         = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_relation.h5"
        self.combined_image_path            = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_img.h5"
        self.combined_node_mapping_path     = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_node2idx.json"
        self.combined_relation_mapping_path = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_relation2idx.json"
        self.combined_image_mapping_path    = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_img2idx.json"

        # since entity embedding's index is same in all dataset, we can use same index for all dataset
        self.G_entity_head_path             = f"{self.args.base_dir}/embedding/entity/{self.args.dataset}_head_idx.json"
        self.G_entity_relation_path         = f"{self.args.base_dir}/embedding/entity/{self.args.dataset}_relation_idx.json"
        self.G_entity_tail_path             = f"{self.args.base_dir}/embedding/entity/{self.args.dataset}_tail_idx.json"
        self.G_entity_img_path              = f"{self.args.base_dir}/embedding/entity/{self.args.dataset}_img_idx.json"

        self.G_mention_head_path            = f"{self.args.base_dir}/embedding/mention/{self.args.dataset}_{self.args.vlm}_head_idx.json"
        self.G_mention_tail_path            = f"{self.args.base_dir}/embedding/mention/{self.args.dataset}_{self.args.vlm}_tail_idx.json"
        self.G_mention_relation_path        = f"{self.args.base_dir}/embedding/mention/{self.args.dataset}_{self.args.vlm}_relation_idx.json"
        self.G_mention_img_path             = f"{self.args.base_dir}/embedding/mention/{self.args.dataset}_{self.args.vlm}_img_idx.json"
        self.G_mention_sent_path            = f"{self.args.base_dir}/embedding/mention//{self.args.dataset}_{self.args.vlm}_sent_idx.json"


        self.load_json_data()
        self.load_mapping()

        self.embed_entity()
        self.embed_mention()
        self.combine_embed()

        if os.path.exists(self.G_mention_tail_path):
            print("Mention index dictionary already exists")
            with open(self.G_mention_tail_path, 'r') as f:
                self.G_mention_tail = json.load(f)
                self._node_stats(self.G_mention_tail)
            with open(self.G_mention_head_path, 'r') as f:
                self.G_mention_head = json.load(f)
            with open(self.G_mention_relation_path, 'r') as f:
                self.G_mention_relation = json.load(f)
            with open(self.G_mention_img_path, 'r') as f:
                self.G_mention_img = json.load(f)
            with open(self.G_mention_sent_path, 'r') as f:
                self.G_mention_sent = json.load(f)
        else:
            self.process_mention()


        if os.path.exists(self.G_entity_tail_path):
            print("Entity index dictionary already exists")
            with open(self.G_entity_tail_path, 'r') as f:
                self.G_entity_tail = json.load(f)
                self._node_stats(self.G_entity_tail)
            with open(self.G_entity_head_path, 'r') as f:
                self.G_entity_head = json.load(f)
            with open(self.G_entity_relation_path, 'r') as f:
                self.G_entity_relation = json.load(f)
            with open(self.G_entity_img_path, 'r') as f:
                self.G_entity_img = json.load(f)
        else :
            self.process_entity()



    def embed_mention(self):
        # Create and process mention encoders
        if not os.path.exists(self.mention_text_embedding_path):
            mention_text_encoder = MentionTextEncoder(  mapping_path=self.mention_text_mapping_path, 
                                                        embedding_path=self.mention_text_embedding_path)
            mention_text_encoder.process(self.json_data, batch_size= self.args.encoding_batch_size)

        if not os.path.exists(self.mention_img_embedding_path):
            mention_img_encoder = MentionImageEncoder(embedding_path=self.mention_img_embedding_path,
                                                      img_dir = f"{self.args.data_dir}/dataset/image/{self.args.dataset}")
            mention_img_encoder.process(self.json_data, batch_size= self.args.encoding_batch_size)

    def embed_entity(self):
        # Create and process entity encoders
        if not os.path.exists(self.entity_text_embedding_path) :
            entity_text_encoder = EntityTextEncoder( embedding_path=self.entity_text_embedding_path )
            entity_text_encoder.process(batch_size= self.args.encoding_batch_size)

        if not os.path.exists(self.entity_img_embedding_path):
            entity_img_encoder = EntityImageEncoder( embedding_path=self.entity_img_embedding_path,
                                                     img_dir = f"{self.args.data_dir}/KB/image")    
            entity_img_encoder.process(batch_size= self.args.encoding_batch_size)
    
    def combine_embed(self):
        """
        Combine entity and mention embeddings using the CombineEmbeddings class.
        This method loads embeddings from individual files and combines them into
        unified embeddings for nodes, relations, and images.
        """

        # Initialize CombineEmbeddings with all necessary paths
        combine_embedding = CombineEmbeddings(
            # Input paths for entity embeddings
            entity_text_path            = self.entity_text_embedding_path,
            entity_img_path             = self.entity_img_embedding_path,
            
            # Input paths for mention embeddings
            mention_text_path           = self.mention_text_embedding_path,
            mention_text_mapping_path   = self.mention_text_mapping_path,
            mention_img_path            = self.mention_img_embedding_path,
            
            # Output paths for combined embeddings
            node_path                   = self.combined_node_path,
            relation_path               = self.combined_relation_path,
            image_path                  = self.combined_image_path,
            
            # Output paths for mapping files
            node_mapping_path           = self.combined_node_mapping_path,
            relation_mapping_path       = self.combined_relation_mapping_path,
            image_mapping_path          = self.combined_image_mapping_path,
        )
        
        # Combine embeddings and get results
        self.node_embeddings, self.node2idx     = combine_embedding.combine_node_embeddings()
        self.rel_embeddings, self.relation2idx  = combine_embedding.combine_relation_embeddings()
        self.img_embeddings, self.img_node2idx  = combine_embedding.combine_image_embeddings()
        
        print(f"Embedding combination completed for dataset: {self.args.dataset}")
        print(f"Node embeddings shape: {self.node_embeddings.shape}")
        print(f"Relation embeddings shape: {self.rel_embeddings.shape}")
        print(f"Image embeddings shape: {self.img_embeddings.shape}")


    def load_json_data(self):
        with open(self.json_path, 'r') as f:
            self.json_data = json.load(f)
        print(f"Loaded {len(self.json_data)} items from {self.json_path}")


    def load_mapping(self):
        """Load dataset split IDs"""
        split_path          = f"{self.args.data_dir}/dataset/mapping/ids_split_mappings.json"
        head_candidate_path = f"{self.args.data_dir}/dataset/mapping/qids_candidate.json"
        blank_img_path      = f"{self.args.data_dir}/dataset/mapping/ids_blank_image.json"
        img_path            = f"{self.args.data_dir}/dataset/mapping/ids_image.json"
        with open(split_path, 'r') as f:
            split2id = json.load(f)
        self.train_ids, self.val_ids, self.test_ids = split2id[self.args.dataset]['train'], split2id[self.args.dataset]['val'], split2id[self.args.dataset]['test']
        print(f"Loaded {len(self.train_ids)} train, {len(self.val_ids)} val, {len(self.test_ids)} test sentence IDs")

        with open(head_candidate_path, 'r') as f:
            data = json.load(f)
        self.head_qid_lst = data[self.args.dataset]
        print(f"Loaded {len(self.head_qid_lst)} head candidates for {self.args.dataset}")

        with open(img_path, 'r') as f:
            data = json.load(f)
        self.img_lst = data[self.args.dataset]
        print(f"Loaded {len(self.img_lst)} image candidates for {self.args.dataset}")

        # for using zero image embedding for blank images
        with open(blank_img_path, 'r') as f:
            data = json.load(f)
        self.blank_img_lst = data[self.args.dataset]
        print(f"Loaded {len(self.blank_img_lst)} blank image id for {self.args.dataset}")

        qid_label_path = f"{self.args.data_dir}/KB/QID2Label.tsv"
        pid_label_path = f"{self.args.data_dir}/KB/PID2Label.tsv" 
        triple_path = f"{self.args.data_dir}/KB/Triples-{self.args.dataset}.tsv" 
    
        # Use names instead of header to specify column names
        self.qid2label = pd.read_csv(qid_label_path, sep='\t', names=['qid', 'label', 'desc'])
        self.pid2label = pd.read_csv(pid_label_path, sep='\t', names=['pid', 'label', 'desc'])

        # Convert to dictionary mapping id to label
        self.qid2label = self.qid2label.set_index('qid')['label'].to_dict()
        self.pid2label = self.pid2label.set_index('pid')['label'].to_dict()
        self.label2pid = {str(v).lower(): k for k, v in self.pid2label.items()}
        self.label2qid = {str(v).lower(): k for k, v in self.qid2label.items()}
        print(f"Loaded {len(self.qid2label)} QID mappings and {len(self.pid2label)} PID mappings")

        self.triples = pd.read_csv(triple_path, sep='\t', names=['head', 'relation', 'tail'])
        print(f"Loaded {len(self.triples)} triples from {triple_path}")


    def process_entity(self):

        

        self.G_entity_tail = {}
        self.G_entity_head = {}
        self.G_entity_img = {}
        self.G_entity_relation = {}

        node_list = list(self.qid2label.keys())

        # During initialization, create a dictionary mapping heads to their triples
        self.head_groups = self.triples.groupby('head')


        for head_qid in tqdm(self.head_qid_lst, desc="Processing entities in KB"):
            if head_qid not in node_list:
                continue
                
            self.G_entity_head[head_qid] = self.get_node_idx(head_qid)
            self.G_entity_img[head_qid] = self.get_entity_img_idx(head_qid)

            # Get triples with head_qid as head
            try:
                group = self.head_groups.get_group(head_qid)
                self.G_entity_tail[head_qid] = [self.get_node_idx(tail) for tail in group['tail']]
                self.G_entity_relation[head_qid] = [self.get_relation_idx(rel) for rel in group['relation']]
            # Skip if head_qid has no triples
            except KeyError:
                self.G_entity_tail[head_qid] = []
                self.G_entity_relation[head_qid] = []


        # Filter out triples with None values
        for head_qid in self.G_entity_head.keys():
            self.G_entity_tail[head_qid], self.G_entity_relation[head_qid] = self._filter_valid_triples(
                self.G_entity_head[head_qid],
                self.G_entity_tail[head_qid],
                self.G_entity_relation[head_qid]
            )


        print(f"Processed {len( self.G_entity_tail)} ENTITY")
        self._node_stats(self.G_entity_tail)
        self._img_stats(self.G_entity_img)

        assert len(self.G_entity_tail) == len(self.G_entity_head) == len(self.G_entity_relation)  == len(self.G_entity_img)

        with open(self.G_entity_tail_path, 'w') as f:
            json.dump(self.G_entity_tail, f)

        with  open(self.G_entity_head_path, 'w') as f:
            json.dump(self.G_entity_head, f)

        with open(self.G_entity_relation_path, 'w') as f:
            json.dump(self.G_entity_relation, f)

        with open(self.G_entity_img_path, 'w') as f:
            json.dump(self.G_entity_img, f)
    

    def process_mention(self):
        
        print(f"Loaded {len(self.json_data)} items for {self.args.dataset}")

        self.G_mention_tail         = {"train": {}, "val": {}, "test": {}}
        self.G_mention_head         = {"train": {}, "val": {}, "test": {}}
        self.G_mention_relation     = {"train": {}, "val": {}, "test": {}}
        self.G_mention_img          = {"train": {}, "val": {}, "test": {}}
        self.G_mention_sent         = {"train": {}, "val": {}, "test": {}}

        for item in tqdm(self.json_data, desc="Processing mentions"):
            for m, answer in zip(item['mention'], item['answer']):
                id = item['id']
                sent = item['sentence']

                desc =  item['desc'].get(m, None)
                desc = '' if desc is None else desc     # if desc is None, set it to empty string
                head = f"{m}: {desc}".strip()

                triple = item['triple'].get(m, [])

                relation = [ t[1].strip() for t in triple] if triple else []
                tail = [t[2].strip() for t in triple] if triple else []

                rel_idx = [self.get_relation_idx(r) for r in relation]
                tail_idx = [self.get_node_idx(t) for t in tail]
                
                if id in self.train_ids:
                    self.G_mention_tail["train"][f"{id}-{answer}"] = tail_idx
                    self.G_mention_relation["train"][f"{id}-{answer}"] = rel_idx
                    self.G_mention_head["train"][f"{id}-{answer}"] = self.get_node_idx(head)
                    self.G_mention_sent["train"][f"{id}-{answer}"] = self.get_node_idx(sent)
                    self.G_mention_img["train"][f"{id}-{answer}"] = self.get_mention_img_idx(id)


                if id in self.val_ids:
                    self.G_mention_tail["val"][f"{id}-{answer}"] = tail_idx
                    self.G_mention_relation["val"][f"{id}-{answer}"] = rel_idx
                    self.G_mention_head["val"][f"{id}-{answer}"] = self.get_node_idx(head)
                    self.G_mention_sent["val"][f"{id}-{answer}"] = self.get_node_idx(sent)
                    self.G_mention_img["val"][f"{id}-{answer}"] = self.get_mention_img_idx(id)
                    

                if id in self.test_ids:
                    self.G_mention_tail["test"][f"{id}-{answer}"] = tail_idx
                    self.G_mention_relation["test"][f"{id}-{answer}"] = rel_idx
                    self.G_mention_head["test"][f"{id}-{answer}"] = self.get_node_idx(head)
                    self.G_mention_sent["test"][f"{id}-{answer}"] = self.get_node_idx(sent)
                    self.G_mention_img["test"][f"{id}-{answer}"] = self.get_mention_img_idx(id)
                    

        # Filter out triples with None values
        for split in ['train', 'val', 'test']:
            for mention_id in self.G_mention_tail[split].keys():
                self.G_mention_tail[split][mention_id], self.G_mention_relation[split][mention_id] = self._filter_valid_triples(
                    self.G_mention_head[split][mention_id],
                    self.G_mention_tail[split][mention_id],
                    self.G_mention_relation[split][mention_id]
                )

        print(f"Processed {len(self.G_mention_tail['train'])} train, {len(self.G_mention_tail['val'])} val, {len(self.G_mention_tail['test'])} test mention ")
        self._node_stats(self.G_mention_tail)
        self._img_stats(self.G_mention_img)

        assert len(self.G_mention_tail['train']) == len(self.G_mention_head['train']) == len(self.G_mention_relation['train'])  == len(self.G_mention_img['train'])

        with open(self.G_mention_tail_path, 'w') as f:
            json.dump(self.G_mention_tail, f)
        with  open(self.G_mention_head_path, 'w') as f:
            json.dump(self.G_mention_head, f)
        with open(self.G_mention_relation_path, 'w') as f:
            json.dump(self.G_mention_relation, f)
        with open(self.G_mention_img_path, 'w') as f:
            json.dump(self.G_mention_img, f)
        with open(self.G_mention_sent_path, 'w') as f:
            json.dump(self.G_mention_sent, f)

    def get_node_idx(self, node):
        # Skip if node is None
        if not node :
            return None
        return self.node2idx.get(node.strip(), None)
    
    def get_relation_idx(self, relation):
        # Skip if relation is None
        if not relation:
            return None
        return self.relation2idx.get(relation.strip(), None)
    
    def get_mention_img_idx(self, id):
        # Skip if image is blank
        if id in self.blank_img_lst:
            return None
        return self.img_node2idx.get(f"mention-{id}", None)
    
    def get_entity_img_idx(self, id):
        # Skip if image is blank
        if f"{id}_0" not in self.img_lst:
            return None
        return self.img_node2idx.get(f"{id}_0", None)

    def _filter_valid_triples(self, head, tails, relations):
        """Helper function to filter out triples with None values."""
        if head is None :
            return [], []
        
        valid_pairs = [(t, r) for t, r in zip(tails, relations) if t is not None and r is not None]

        if not valid_pairs:
            return [], []

        filtered_tails, filtered_relations = zip(*valid_pairs)

        return list(filtered_tails), list(filtered_relations)

    def _node_stats(self, node_list):
        if len(node_list) == 3: # train,test,val
            # merge three dictionaries
            node_list = node_list['train'] | node_list['val'] | node_list['test']
        avg = 0
        for node in node_list.keys():
            avg += len(node_list[node])
        print(f"Avg. # of nodes : {avg / len(node_list):.2f} | STD. # of nodes : {np.std([len(v) for v in node_list.values()]):.2f} | Max # of nodes : {max([len(v) for v in node_list.values()])}  |  Min # of nodes : {min([len(v) for v in node_list.values()])} ")

    def _img_stats(self, img_list):
        if len(img_list) == 3:
            img_list = img_list['train'] | img_list['val'] | img_list['test']
        total =[ 1 for img in img_list.keys() if img_list[img] is not None]
        print(f"Total # of images : {len(total)} ")


class CombineEmbeddings:
    """Process and combine entity and mention embeddings"""
    
    def __init__(self, 
                 entity_text_path: str,
                 entity_img_path: str,
                 mention_text_path: str,
                 mention_text_mapping_path: str,
                 mention_img_path: str,
                 node_path: str,
                 relation_path: str,
                 image_path: str,
                 node_mapping_path: str = None,
                 relation_mapping_path: str = None,
                 image_mapping_path: str = None,
                 device: str = 'cpu'):
        """
        Initialize the embedding combiner.
        
        Args:
            entity_text_path: Path to entity text embeddings file
            entity_img_path: Path to entity image embeddings file
            mention_text_path: Path to mention text embeddings file
            mention_text_mapping_path: Path to mention text mapping file
            mention_img_path: Path to mention image embeddings file
            node_path: Output path for combined node embeddings
            relation_path: Output path for combined relation embeddings
            image_path: Output path for combined image embeddings
            node_mapping_path: Output path for node mapping file (optional)
            relation_mapping_path: Output path for relation mapping file (optional)
            image_mapping_path: Output path for image mapping file (optional)
            device: Device to load tensors to ('cpu' or 'cuda')
        """
        self.device = device
        
        # Input paths
        self.entity_text_embedding_path = entity_text_path
        self.entity_img_embedding_path = entity_img_path
        self.mention_text_embedding_path = mention_text_path
        self.mention_text_mapping_path = mention_text_mapping_path
        self.mention_img_embedding_path = mention_img_path
        
        # Output paths
        self.combined_node_path = node_path
        self.combined_relation_path = relation_path
        self.combined_image_path = image_path
        
        # Derive output mapping paths from combined paths if not provided
        node_dir = os.path.dirname(node_path)
        self.node_mapping_path = node_mapping_path or f"{node_dir}/node2idx.json"
        self.relation_mapping_path = relation_mapping_path or f"{node_dir}/relation2idx.json"
        self.image_mapping_path = image_mapping_path or f"{node_dir}/img2idx.json"
        
        # Load entity and mention embeddings
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load all embeddings and mappings"""
        # Load entity embeddings
        with h5py.File(self.entity_text_embedding_path, 'r') as f:
            self.entity_text_embeddings = torch.tensor(f['embeddings'][()]).to(self.device)
            entity_ids = [id.decode() if isinstance(id, bytes) else id for id in f['ids'][()]]
        
        # Separate entity and relation embeddings
        entity_idx = [idx for idx, id_ in enumerate(entity_ids) if str(id_).startswith('Q')]
        entity_qids = [entity_ids[idx] for idx in entity_idx]
        self.entity_node_qid2idx = {id: idx for idx, id in enumerate(entity_qids)}
        self.entity_node_embeddings = self.entity_text_embeddings[entity_idx]
        
        relation_idx = [idx for idx, id_ in enumerate(entity_ids) if str(id_).startswith('P')]
        relation_pids = [entity_ids[idx] for idx in relation_idx]
        self.entity_relation_pid2idx = {id: idx for idx, id in enumerate(relation_pids)}
        self.entity_relation_embeddings = self.entity_text_embeddings[relation_idx]
        
        # Load entity image embeddings
        with h5py.File(self.entity_img_embedding_path, 'r') as f:
            self.entity_img_embeddings = torch.tensor(f['embeddings'][()]).to(self.device)
            img_ids = [id.decode() if isinstance(id, bytes) else id for id in f['ids'][()]]
        self.entity_img2idx = {id: idx for idx, id in enumerate(img_ids)}
        
        # Load mention embeddings
        with h5py.File(self.mention_text_embedding_path, 'r') as f:
            self.mention_text_embeddings = f['embeddings'][()]

        # Load mention mappings
        with open(self.mention_text_mapping_path, 'r') as f:
            mention_mappings = json.load(f)
            self.mention_head2idx = mention_mappings['head']
            self.mention_relation2idx = mention_mappings['relation']
            self.mention_tail2idx = mention_mappings['tail']
            self.mention_sentence2idx = mention_mappings['sentence']
        
        # Load mention image embeddings
        with h5py.File(self.mention_img_embedding_path, 'r') as f:
            self.mention_img_embeddings = f['embeddings'][()]
            mention_img_ids = [id.decode() if isinstance(id, bytes) else id for id in f['ids'][()]]
        self.mention_img2idx = {f"mention-{id}": idx for idx, id in enumerate(mention_img_ids)}

        print("All embeddings and mappings loaded successfully")
    
    def combine_node_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """
        Combine entity and mention node embeddings
        
        Returns:
            Tuple containing combined node embeddings and mapping dictionary
        """
        if os.path.exists(self.combined_node_path) and os.path.exists(self.node_mapping_path):
            with h5py.File(self.combined_node_path, 'r') as f:
                self.combined_node_embeddings = f['embeddings'][()]
                    
            with open(self.node_mapping_path, 'r') as f:   
                self.node2idx = json.load(f)
            print("Combined node embeddings and mapping already exist")
            return self.combined_node_embeddings, self.node2idx
            
        # Get indices for different types of mention embeddings
        head_start_idx = min(self.mention_head2idx.values())
        head_end_idx = max(self.mention_head2idx.values())
        tail_start_idx = min(self.mention_tail2idx.values())
        tail_end_idx = max(self.mention_tail2idx.values())
        sent_start_idx = min(self.mention_sentence2idx.values())
        sent_end_idx = max(self.mention_sentence2idx.values())
        
        # Concatenate embeddings
        self.combined_node_embeddings = np.concatenate([
            self.entity_node_embeddings,
            self.mention_text_embeddings[head_start_idx:head_end_idx+1],
            self.mention_text_embeddings[tail_start_idx:tail_end_idx+1],
            self.mention_text_embeddings[sent_start_idx:sent_end_idx+1]
        ])
        print(f"Combined node embeddings shape: {self.combined_node_embeddings.shape}")
        
        # Calculate offsets for index mapping
        head_offset = self.entity_node_embeddings.shape[0]
        tail_offset = self.mention_text_embeddings[head_start_idx:head_end_idx+1].shape[0] + head_offset
        sent_offset = self.mention_text_embeddings[tail_start_idx:tail_end_idx+1].shape[0] + tail_offset
        
        # Create combined mapping
        self.node2idx = {
            **self.entity_node_qid2idx,
            **{k: v - head_start_idx + head_offset for k, v in self.mention_head2idx.items()},
            **{k: v - tail_start_idx + tail_offset for k, v in self.mention_tail2idx.items()},
            **{k: v - sent_start_idx + sent_offset for k, v in self.mention_sentence2idx.items()}
        }
        
        # Save combined embeddings and mapping
        os.makedirs(os.path.dirname(self.combined_node_path), exist_ok=True)
        with h5py.File(self.combined_node_path, 'w') as f:
            f.create_dataset('embeddings', data=self.combined_node_embeddings, compression='gzip')
        
        with open(self.node_mapping_path, 'w') as f:
            json.dump(self.node2idx, f)
        
        print(f"Combined node embeddings saved to {self.combined_node_path}")
        print(f"Node mapping saved to {self.node_mapping_path}")
        
        return self.combined_node_embeddings, self.node2idx
    
    def combine_relation_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """
        Combine entity and mention relation embeddings
        
        Returns:
            Tuple containing combined relation embeddings and mapping dictionary
        """
        if os.path.exists(self.combined_relation_path) and os.path.exists(self.relation_mapping_path):
            with h5py.File(self.combined_relation_path, 'r') as f:
                self.combined_relation_embeddings = f['embeddings'][()]

            with open(self.relation_mapping_path, 'r') as f:   
                self.relation2idx = json.load(f)
            print("Combined relation embeddings and mapping already exist")
            return self.combined_relation_embeddings, self.relation2idx
        
        # Get indices for mention relation embeddings
        rel_start_idx = min(self.mention_relation2idx.values())
        rel_end_idx = max(self.mention_relation2idx.values())
        
        # Calculate offset
        offset = self.entity_relation_embeddings.shape[0]
        
        # Concatenate embeddings
        self.combined_relation_embeddings = np.concatenate([
            self.entity_relation_embeddings,
            self.mention_text_embeddings[rel_start_idx:rel_end_idx+1]
        ])
        print(f"Combined relation embeddings shape: {self.combined_relation_embeddings.shape}")
        
        # Create combined mapping
        self.relation2idx = {
            **self.entity_relation_pid2idx,
            **{k: v - rel_start_idx + offset for k, v in self.mention_relation2idx.items()}
        }
        
        # Save combined embeddings and mapping
        os.makedirs(os.path.dirname(self.combined_relation_path), exist_ok=True)
        with h5py.File(self.combined_relation_path, 'w') as f:
            f.create_dataset('embeddings', data=self.combined_relation_embeddings, compression='gzip')
        
        with open(self.relation_mapping_path, 'w') as f:
            json.dump(self.relation2idx, f)
        
        print(f"Combined relation embeddings saved to {self.combined_relation_path}")
        print(f"Relation mapping saved to {self.relation_mapping_path}")
        
        return self.combined_relation_embeddings, self.relation2idx
    
    def combine_image_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """
        Combine entity and mention image embeddings
        
        Returns:
            Tuple containing combined image embeddings and mapping dictionary
        """
        if os.path.exists(self.combined_image_path) and os.path.exists(self.image_mapping_path):
            with h5py.File(self.combined_image_path, 'r') as f:
                self.combined_img_embeddings = f['embeddings'][()]

            with open(self.image_mapping_path, 'r') as f:   
                self.img2idx = json.load(f)
            print("Combined image embeddings and mapping already exist")
            return self.combined_img_embeddings, self.img2idx
        
        # Calculate offset
        offset = self.entity_img_embeddings.shape[0]
        
        # Concatenate embeddings
        self.combined_img_embeddings = np.concatenate([
            self.entity_img_embeddings,
            self.mention_img_embeddings
        ])
        print(f"Combined image embeddings shape: {self.combined_img_embeddings.shape}")

        # Create combined mapping
        self.img2idx = {
            **self.entity_img2idx,
            **{k: v + offset for k, v in self.mention_img2idx.items()}
        }

        # Save combined embeddings and mapping
        os.makedirs(os.path.dirname(self.combined_image_path), exist_ok=True)
        with h5py.File(self.combined_image_path, 'w') as f:
            f.create_dataset('embeddings', data=self.combined_img_embeddings, compression='gzip')
        
        with open(self.image_mapping_path, 'w') as f:
            json.dump(self.img2idx, f)
        
        print(f"Combined image embeddings saved to {self.combined_image_path}")
        print(f"Image mapping saved to {self.image_mapping_path}")
        
        return self.combined_img_embeddings, self.img2idx



# def parse_args():
#     parser = argparse.ArgumentParser(description="Embedding Pipeline")
#     parser.add_argument("--data_dir", type=str, default="/workspace/KGMEL/data", help="Data directory")
#     parser.add_argument("--base_dir", type=str, default="/workspace/KGMEL", help="Base directory")
#     parser.add_argument("--dataset", type=str, default="WikiMEL", help="Dataset name")
#     parser.add_argument("--vlm", type=str, default="gpt-4o-mini-2024-07-18", help="VLM model name")
#     parser.add_argument("--encoding_batch_size", type=int, default=2048, help="Batch size for processing")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     EmbeddingProcessor(args)

