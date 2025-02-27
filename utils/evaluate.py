import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def generate_entity_embeddings(model, entity_loader, device):
    """
    Generate embeddings for all entities.
    
    Args:
        model: The neural network model to use for embedding generation
        entity_loader: DataLoader providing entity batches
        device: Device to perform computations on ('cpu' or 'cuda')
        
    Returns:
        tuple: (entity_embeddings, entity_ids)
            - entity_embeddings: Tensor of normalized entity embeddings
            - entity_ids: List of entity QIDs corresponding to the embeddings
    """
    model.eval()
    entity_lst = []
    qids = []
    
    # Process batches
    for batch in tqdm(entity_loader, desc="Generating Entity Embeddings", ncols=100):
        # Move tensors to device
        head_batch = batch['head_emb'].to(device)
        rel_batch = batch['relation_emb'].to(device)
        tail_batch = batch['tail_emb'].to(device)
        entity_mask = batch['entity_mask'].to(device)
        img_batch = batch['img_emb'].to(device)
        qids.extend(batch['qid'])
        
        # Generate entity representations using the model
        entity = model(
            x_head=head_batch, 
            x_rel=rel_batch,
            x_tail=tail_batch, 
            x_mask=entity_mask, 
            x_img=img_batch
        )
        
        # Normalize embeddings for cosine similarity
        entity = F.normalize(entity, p=2, dim=1)
        entity_lst.append(entity)
        
    # Concatenate all entity embeddings
    return torch.cat(entity_lst, dim=0), qids


@torch.no_grad()
def evaluate(model, val_loader, test_loader, entity_loader, 
                              device='cuda', k_values=[1, 3, 5, 10, 16, 100]):
    """
    Evaluate model performance on validation and test sets.
    
    Args:
        model: The neural network model to evaluate
        val_loader: DataLoader providing validation data
        test_loader: DataLoader providing test data
        entity_loader: DataLoader providing entity data
        device: Device to perform computations on ('cpu' or 'cuda')
        k_values: List of k values for Hits@k metric calculation
        
    Returns:
        tuple: (val_hits, val_mrr, test_hits, test_mrr)
            - val_hits: Dictionary of Hits@k values for validation set
            - val_mrr: Mean Reciprocal Rank for validation set
            - test_hits: Dictionary of Hits@k values for test set
            - test_mrr: Mean Reciprocal Rank for test set
    """
    model.eval()
    
    # Get entity embeddings
    entity, qids = generate_entity_embeddings(
        model=model, 
        entity_loader=entity_loader, 
        device=device
    )
    
    # Create mapping from entity IDs to indices
    entity_qids_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    result = {}
    
    # Evaluate on validation and test sets
    for split, data_loader in [("Val", val_loader), ("Test", test_loader)]:
        eval_progress = tqdm(data_loader, desc=f"Evaluating {split}", ncols=100)
        hits, mrr, total = {k: 0 for k in k_values}, 0, 0
        
        for batch in eval_progress:
            # Get mention embeddings
            mention_head_emb = batch['mention_head_emb'].to(device)
            mention_rel_emb = batch['mention_relation_emb'].to(device)
            mention_tail_emb = batch['mention_tail_emb'].to(device)
            mention_mask = batch['mention_mask'].to(device)
            mention_img_emb = batch['mention_img_emb'].to(device)
            
            # Forward pass to get mention representations
            mention = model(
                x_head=mention_head_emb,
                x_rel=mention_rel_emb,
                x_tail=mention_tail_emb,
                x_mask=mention_mask,
                x_img=mention_img_emb
            )
            
            # Normalize mention embeddings for cosine similarity
            mention = F.normalize(mention, p=2, dim=1)
            
            # Compute similarity between mentions and all entities
            similarity = torch.mm(mention, entity.T)

            # Compute similarity using consine similarity
            # similarity =  F.cosine_similarity(mentions.unsqueeze(1), entity.unsqueeze(0), dim=2)
            
            # Get true entity indices and compute ranks
            true_indices = torch.tensor([entity_qids_to_idx[qid] for qid in batch['answer']], device=device)
            ranks = torch.where(
                torch.argsort(similarity, dim=1, descending=True) == true_indices.unsqueeze(1)
            )[1] + 1
            
            # Update metrics
            current_mrr = (1.0 / ranks).mean().item()
            mrr += (1.0 / ranks).sum().item()
            
            for k in k_values:
                hits[k] += (ranks <= k).sum().item()
            total += len(true_indices)
            
            # Update progress bar with current metrics
            eval_progress.set_postfix({
                'MRR': f'{current_mrr*100:.2f}',
                'H@1': f"{hits[1]*100/total:.2f}",
                "H@3": f"{hits[3]*100/total:.2f}",
            })
        
        # Compute final metrics
        hits = {k: v/total for k, v in hits.items()}
        mrr = mrr/total
        
        # Print results
        print(f"{split} Results:", end="\t")
        for k, v in hits.items():
            print(f"Hits@{k}: {v*100:.2f}", end="\t")
        print(f"MRR: {mrr*100:.2f}") 
        
        result[split] = {"hits": hits, "mrr": mrr}
    
    return (result['Val']['hits'], result['Val']['mrr'], 
            result['Test']['hits'], result['Test']['mrr'])


@torch.no_grad()
def generate_candidate_preds(model, train_loader, val_loader, test_loader,
                                  entity_loader, device, k_values=[1, 3, 5, 10, 16, 100],
                                  num_candidate=16):
    """
    Generate candidate predictions for all datasets and evaluate performance.
    
    Args:
        model: The neural network model to use
        train_loader: DataLoader providing training data
        val_loader: DataLoader providing validation data
        test_loader: DataLoader providing test data
        entity_loader: DataLoader providing entity data
        device: Device to perform computations on ('cpu' or 'cuda')
        k_values: List of k values for Hits@k metric calculation
        num_candidate: Number of candidates to generate for each mention
        
    Returns:
        tuple: (train_predictions, val_predictions, test_predictions)
            - Dictionaries containing prediction data for each split
    """
    model.eval()
    
    # Generate entity embeddings once
    entity, qids = generate_entity_embeddings(
        model=model, 
        entity_loader=entity_loader, 
        device=device
    )
    
    # Create mapping from entity IDs to indices
    entity_qids_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    
    # Initialize result dictionaries
    train_predictions, val_predictions, test_predictions = {}, {}, {}
    
    # Process each data split
    for split, data_loader, predictions in [
        ('Train', train_loader, train_predictions), 
        ('Val', val_loader, val_predictions), 
        ('Test', test_loader, test_predictions)
    ]:
        eval_progress = tqdm(data_loader, desc=f"Evaluating {split}", ncols=100)
        hits, mrr, total = {k: 0 for k in k_values}, 0, 0
        
        # Initialize prediction lists
        predictions['answer'] = []
        predictions['mention_key'] = []
        predictions['candidate'] = []
        predictions['rank'] = []
        
        for batch in eval_progress:
            # Get mention data
            mention_head_emb = batch['mention_head_emb'].to(device)
            mention_rel_emb = batch['mention_relation_emb'].to(device)
            mention_tail_emb = batch['mention_tail_emb'].to(device)
            mention_mask = batch['mention_mask'].to(device)
            mention_img_emb = batch['mention_img_emb'].to(device)
            
            mention_key = batch['mention_key']
            answer = batch['answer']
            
            # Store ground truth and mention keys
            predictions['answer'].extend(answer)
            predictions['mention_key'].extend(mention_key)
            
            # Generate mention embeddings
            mentions = model(
                x_head=mention_head_emb,
                x_rel=mention_rel_emb,
                x_tail=mention_tail_emb,
                x_mask=mention_mask,
                x_img=mention_img_emb
            )
            
            # Normalize for cosine similarity
            mentions = F.normalize(mentions, p=2, dim=1)
            
            # Compute similarity to all entities
            similarity = torch.mm(mentions, entity.T)

            # Compute similarity using consine similarity
            # similarity =  F.cosine_similarity(mentions.unsqueeze(1), entity.unsqueeze(0), dim=2)
            
            # Get top-k predictions
            _, top_k_indices = torch.topk(similarity, k=num_candidate, dim=1)
            
            # Convert indices to entity IDs
            for indices in top_k_indices:
                predictions['candidate'].append([qids[idx.item()] for idx in indices])
            
            # Calculate metrics
            true_indices = torch.tensor([entity_qids_to_idx[qid] for qid in batch['answer']], device=device)
            ranks = torch.where(
                torch.argsort(similarity, dim=1, descending=True) == true_indices.unsqueeze(1)
            )[1] + 1
            
            predictions['rank'] += ranks.tolist()
            
            # Update evaluation metrics
            current_mrr = (1.0 / ranks).mean().item()
            mrr += (1.0 / ranks).sum().item()
            
            for k in k_values:
                hits[k] += (ranks <= k).sum().item()
            total += len(true_indices)
            
            # Update progress information
            eval_progress.set_postfix({
                'MRR': f'{100*current_mrr:.2f}',
                'Hit@1': f"{100*hits[1]/total:.2f}",
                'Hit@3': f"{100*hits[3]/total:.2f}",
            })
        
        # Print final results
        print(f"{split} Results:", end="\t")
        for k, v in hits.items():
            print(f"Hits@{k}: {100*v/total:.2f}", end="\t")
        print(f"MRR: {100*mrr/total:.2f}")
    
    return train_predictions, val_predictions, test_predictions