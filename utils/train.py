
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



def train_epoch(model, loader, optimizer, device, temperature=0.2, batch_size=32, lambda_mm=0.1, lambda_ee=0.1):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model to train
        loader: DataLoader providing batched training data
        optimizer: Optimizer for parameter updates
        device: Device to perform computations on ('cpu' or 'cuda')
        temperature: Temperature parameter for contrastive loss
        batch_size: Batch size for training
        lambda_mm: Weight for mention-mention contrastive loss
        lambda_ee: Weight for entity-entity contrastive loss
        
    Returns:
        float: Average loss value for the epoch
    """
    model.train()
    total_loss = []
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", ncols=100)
    lambda_mm = torch.tensor(lambda_mm, device=device)
    lambda_ee = torch.tensor(lambda_ee, device=device)
    
    for i, batch in progress_bar:
        # Move all tensors to the specified device
        mention_head_emb = batch['mention_head_emb'].to(device)  # [batch_size, embed_dim]
        mention_rel_emb = batch['mention_relation_emb'].to(device)  # [batch_size, max_mention_relations, embed_dim]
        mention_tail_emb = batch['mention_tail_emb'].to(device)  # [batch_size, max_mention_tails, embed_dim]
        mention_mask = batch['mention_mask'].to(device)  # [batch_size, max_seq_len]
        mention_img_emb = batch['mention_img_emb'].to(device)  # [batch_size, embed_dim]

        pos_head_emb = batch['positive_head_emb'].to(device)  # [batch_size, embed_dim]
        pos_rel_emb = batch['positive_relation_emb'].to(device)  # [batch_size, max_pos_relations, embed_dim]
        pos_tail_emb = batch['positive_tail_emb'].to(device)  # [batch_size, max_pos_tails, embed_dim]
        pos_mask = batch['positive_mask'].to(device)  # [batch_size, max_seq_len]
        pos_img_emb = batch['positive_img_emb'].to(device)  # [batch_size, embed_dim]

        # Pass separate relation and tail embeddings to the model
        mention_graphs = model(
            x_head=mention_head_emb,
            x_rel=mention_rel_emb,
            x_tail=mention_tail_emb,
            x_mask=mention_mask,
            x_img=mention_img_emb
        )

        # Get positive entity graph embeddings
        entity_graphs = model(
            x_head=pos_head_emb,
            x_rel=pos_rel_emb,
            x_tail=pos_tail_emb,
            x_mask=pos_mask,
            x_img=pos_img_emb
        )

        # Normalize embeddings for contrastive loss
        mention_graphs = F.normalize(mention_graphs, p=2, dim=-1)
        entity_graphs = F.normalize(entity_graphs, p=2, dim=-1)

        # Calculate contrastive losses
        loss = cl_loss(mention_graphs, entity_graphs, temperature)
        loss_mm = cl_loss(mention_graphs, mention_graphs, temperature)
        loss_ee = cl_loss(entity_graphs, entity_graphs, temperature)

        # Combine losses with weights
        total_loss_val = lambda_mm * loss_mm + lambda_ee * loss_ee + loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss_val.backward()
        optimizer.step()

        # Track loss
        current_loss = total_loss_val.item()
        total_loss.append(current_loss)
        progress_bar.set_postfix({'TRAIN LOSS': f'{np.mean(total_loss):.4f}'})

    # Calculate average loss over the epoch
    avg_loss = np.mean(total_loss)
    return avg_loss


def cl_loss(pos, aug, temperature=0.2):
    """
    Compute contrastive loss between positive and augmented embeddings.
    Args:
        pos (torch.Tensor): First embedding set of shape [batch_size, embed_dim]
        aug (torch.Tensor): Second embedding set of shape [batch_size, embed_dim]
        temperature (float): Temperature parameter controlling distribution sharpness
                            Lower values make the distribution more concentrated
    
    Returns:
        torch.Tensor: Scalar contrastive loss value
    """
    # Normalize embeddings to unit length (L2 norm)
    pos = F.normalize(pos, p=2, dim=1)  # [batch_size, embed_dim]
    aug = F.normalize(aug, p=2, dim=1)  # [batch_size, embed_dim]
    
    # Compute dot product between corresponding pairs (positive examples)
    pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
    # Compute dot products between all possible pairs
    ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]
    
    # Alternative implementation using cosine similarity
    # pos_score = cosine_similarity(pos, aug, dim=1)  # [batch_size]
    # ttl_score = cosine_similarity(pos.unsqueeze(1), aug.unsqueeze(0), dim=2)  # [batch_size, batch_size]
    
    # Apply temperature scaling to control the sharpness of the distribution
    pos_score = torch.exp(pos_score / temperature)  # [batch_size]
    ttl_score = torch.sum(torch.exp(ttl_score / temperature), axis=1)  # [batch_size]
    
    # Compute negative log likelihood with epsilon for numerical stability
    c_loss = -torch.mean(torch.log(pos_score / ttl_score + 1e-12))
    
    return c_loss