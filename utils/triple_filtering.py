import json
import os
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def triple_filtering(data, predictions, args):
    """

    Args:
        data: Data loader containing embeddings and graph structures
        predictions: Dictionary containing prediction candidates
        args: Arguments containing configuration parameters
        
    Returns:
        tuple: (filtered_relations, filtered_tails) - Dictionaries mapping mention-entity
              pairs to their relevant relation and tail indices
    """
    triple_filter = TripleDataset(
        M_node=data.node_embeddings,
        M_rel=data.rel_embeddings,
        G_entity_tail=data.G_entity_tail,
        G_entity_relation=data.G_entity_relation,
        G_mention_tail=data.G_mention_tail['test'],
        G_mention_relation=data.G_mention_relation['test'],
        max_triples_rerank=args.max_triples_rerank,
        predictions=predictions,
        top_n=args.n,
        device=args.device,
    )
    
    filtered_relations, filtered_tails = triple_filter.filter()
    return filtered_relations, filtered_tails



class TripleDataset:
    def __init__(
        self,
        M_node, M_rel, G_entity_tail, G_entity_relation, G_mention_tail, G_mention_relation,
        max_triples_rerank, predictions, top_n, device
    ):
        # Convert numpy arrays to tensors
        self.M_node = torch.tensor(M_node)
        self.M_rel = torch.tensor(M_rel)
        
        # Embedding indices for mentions
        self.G_mention_tail = G_mention_tail 
        self.G_mention_relation = G_mention_relation
    
        # Embedding indices for entities
        self.G_entity_tail = G_entity_tail
        self.G_entity_relation = G_entity_relation

        # Predictions and configuration
        self.predictions = predictions
        self.max_triples_rerank = max_triples_rerank
        self.top_n = top_n
        self.device = device

        # Set dimensions
        self.embed_dim = self.M_node.size(1)

        # Zero embeddings
        self.zero_emb = torch.zeros(1, self.embed_dim)

        # Initialize similarity retriever
        self.retriever = SimilarityRetriever(device=self.device)

        # Output containers
        self.filtered_G_tail = {}
        self.filtered_G_relation = {}




    def filter(self):
        """
        Filter triples based on embedding similarity between mention and entity components.
        
        Returns:
            tuple: (filtered_relations, filtered_tails) - Dictionaries with filtered triples
        """
        before_count, after_count = 0, 0
        max_count = 0
        num_candidates = len(self.predictions['candidate'][0])
        num_mentions = len(self.predictions['mention_key'])
        
        progress = tqdm(
            enumerate(zip(
                self.predictions['mention_key'], 
                self.predictions['answer'], 
                self.predictions['candidate']
            )), 
            total=len(self.predictions['mention_key']),
            desc="Evaluating",
            ncols=100
        )

        for idx, (mention_key, answer, candidates) in progress:

            # Get mention embeddings
            mention_tail_emb = self.get_node_embeddings(
                self.G_mention_tail.get(mention_key, [])
            )
            mention_rel_emb = self.get_relation_embeddings(
                self.G_mention_relation.get(mention_key, [])
            )

            # Collect all unique entity tail and relation indices
            all_entity_tail_indices, all_entity_relation_indices = set(), set()
            for candidate in candidates:
                entity_tail_indices = self.G_entity_tail.get(candidate, [])
                entity_relation_indices = self.G_entity_relation.get(candidate, [])
                all_entity_tail_indices.update(entity_tail_indices)
                all_entity_relation_indices.update(entity_relation_indices)
                
            # Convert sets to lists
            all_entity_relation_indices = list(all_entity_relation_indices)
            all_entity_tail_indices = list(all_entity_tail_indices)
            
            # Get entity embeddings
            entity_rel_emb = self.get_relation_embeddings(all_entity_relation_indices)
            entity_tail_emb = self.get_node_embeddings(all_entity_tail_indices)

            # Retrieve most similar relations and tails
            rel_indices = self.retriever.retrieve_most_similar(
                mention_rel_emb, entity_rel_emb, top_n=self.top_n*2
            )
            relevant_relation_indices = [
                all_entity_relation_indices[i] for i in rel_indices
            ] if rel_indices and all_entity_relation_indices else all_entity_relation_indices
            
            tail_indices = self.retriever.retrieve_most_similar(
                mention_tail_emb, entity_tail_emb, top_n=self.top_n
            )
            relevant_tail_indices = [
                all_entity_tail_indices[i] for i in tail_indices
            ] if tail_indices and all_entity_tail_indices else all_entity_tail_indices

            # Filter triples for each candidate
            for entity_id in candidates:
                filtered_G_tail, filtered_relation_indices = [], []
                before_count += len(self.G_entity_tail[entity_id])
                
                # Keep only triples with both relation and tail in the relevant sets
                for relation, tail in zip(
                    self.G_entity_relation[entity_id], 
                    self.G_entity_tail[entity_id]
                ):
                    if relation not in relevant_relation_indices or tail not in relevant_tail_indices:
                        continue
                    filtered_G_tail.append(tail)
                    filtered_relation_indices.append(relation)
                
                # Store filtered indices
                pair_key = f"{mention_key}-{entity_id}"
                self.filtered_G_tail[pair_key] = filtered_G_tail
                self.filtered_G_relation[pair_key] = filtered_relation_indices
                
                # Update statistics
                after_count += len(filtered_G_tail)
                max_count = max(max_count, len(filtered_G_tail))

        # Print statistics
        print(f"Avg. # of Triples Before Filtering: \t{before_count/(num_candidates*num_mentions):.2f}") 
        print(f"Avg. # of Triples After Filtering: \t{after_count/(num_candidates*num_mentions):.2f}")
        print(f"Max # of Triples: {max_count}")

        return self.filtered_G_relation, self.filtered_G_tail
    
    def get_node_embeddings(self, indices):
        """
        Args:
            indices (List[int]): List of node indices to retrieve
            
        Returns:
            torch.Tensor: Node embeddings, or zero embedding if indices is empty
        """
        if len(indices) == 0:
            return self.zero_emb
        
        node_emb = self.M_node[indices].float()
        
        # Sample if exceeding max_triples_rerank
        if node_emb.size(0) > self.max_triples_rerank:
            idx = torch.randperm(node_emb.size(0))[:self.max_triples_rerank]
            node_emb = node_emb[idx]
        
        return node_emb

    def get_relation_embeddings(self, indices):
        """
        Args:
            indices (List[int]): List of relation indices to retrieve
            
        Returns:
            torch.Tensor: Relation embeddings, or zero embedding if indices is empty/None
        """
        if indices is None or len(indices) == 0:
            return self.zero_emb
        
        rel_emb = self.M_rel[indices].float()
        
        # Sample if exceeding max_triples_rerank
        if rel_emb.size(0) > self.max_triples_rerank:
            idx = torch.randperm(rel_emb.size(0))[:self.max_triples_rerank]
            rel_emb = rel_emb[idx]
        
        return rel_emb

    # Load embeddings
class SimilarityRetriever:
    """
    Retrieves top-n most similar items based on embedding similarity.
    """
    def __init__(self, device='cuda'):
        self.device = device


    def retrieve_most_similar(self, query_embeddings, candidate_embeddings, top_n):
        """
        Args:
            query_embeddings: Query embeddings tensor
            candidate_embeddings: Candidate embeddings tensor
            top_n: Number of top results to retrieve
            
        Returns:
            list: Indices of the top-n most similar candidates
        """
        # Ensure top_n doesn't exceed available candidates
        top_n = min(top_n, candidate_embeddings.size(0)) 
        
        # Move tensors to the correct device
        query_embeddings = query_embeddings.to(self.device)
        candidate_embeddings = candidate_embeddings.to(self.device)
        
        # Compute similarity scores
        similarity_scores = torch.mm(query_embeddings, candidate_embeddings.T)
        
        # Get top-n indices
        _, topn_indices = torch.topk(similarity_scores, k=top_n, dim=1)
        
        # Convert to list and remove duplicates
        unique_indices = list(set(topn_indices.cpu().numpy().flatten().tolist()))
        return unique_indices

