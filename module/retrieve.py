import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.seed import seed_everything
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import os
import wandb
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.embedding_processor import EmbeddingProcessor
from utils.dataloader import get_dataloaders, get_entity_dataloaders
from utils.train import train_epoch
from utils.evaluate import evaluate, generate_candidate_preds

# Enable CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class RetrievalModule():
    def __init__(self, args):
        self.args = args
        self.run_name = f"KGMEL-{self.args.dataset}"
            
        # Prepare checkpoint directory
        checkpoint_dir = f"{self.args.base_dir}/checkpoints"
        self.best_model_path    = f"{checkpoint_dir}/{self.run_name}/best_model.pt"
        self.candidate_path     = f"{checkpoint_dir}/{self.run_name}/candidate-{self.args.num_candidates}.json"
        
        self.min_epochs = {'RichpediaMEL': 5, 'WikiMEL': 2,'WikiDiverse': 50}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(f"{checkpoint_dir}/{self.run_name}", exist_ok=True)

        seed_everything(self.args.seed)
        

    def retrieve(self):
        # Load data 
        self.load_data_loader()

        # Train model
        self.train()

        # Generate candidate predictions
        self.pred_candidate()
  

    def load_data_loader(self):
        # Load data
        print('--------- Embedding Processor ---------')
        self.data = EmbeddingProcessor(self.args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.data, self.args)
        self.entity_loader = get_entity_dataloaders(self.data, self.args)

    
    
        # Initialize model
        wandb.init(project=self.args.wandb_project, name=self.run_name, config=self.args, )
        print('--------- Initialize Model ---------')
        self.model = RetrievalModel(hidden_dim  = self.args.hidden_dim,
                                    temperature = self.args.att_temperature, 
                                    beta        = self.args.beta,
                                    p           = self.args.p,
                                    dropout     = self.args.dropout
                                ).to(self.args.device)
                
        # Calculate and log model parameters
        param = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6 
        print(f"Number of trainable parameters: {param:.2f}M ")
        wandb.config.update({ "num_parameters(M)": param, })

    def train(self):
        # Tracking variables
        best_test = {"hits": {}, "mrr": 0, "epoch": 0}
        early_stopping = 0

        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        print('--------- Training ---------')
        # Training loop
        for epoch in range(1, self.args.max_epochs+1):
            print(f"[Epoch {epoch}/{self.args.max_epochs}]")
            
            # Train for one epoch
            train_loss = train_epoch(
                model       = self.model, 
                loader      = self.train_loader, 
                optimizer   = optimizer, 
                device      = self.args.device, 
                temperature = self.args.cl_temperature, 
                batch_size  = self.args.batch_size, 
                lambda_mm   = self.args.lambda_mm, 
                lambda_ee   = self.args.lambda_ee
            )
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
            
            if epoch%3 != 0 or epoch < self.min_epochs[self.args.dataset]:
                continue

            # Evaluate model performance
            val_hits, val_mrr, test_hits, test_mrr = evaluate(
                model=self.model, 
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                entity_loader=self.entity_loader,
                device=self.args.device
            )
            
            # Log validation and test metrics
            wandb.log({ "epoch": epoch + 1, "val_mrr": val_mrr, **{f"val_hits@{k}": v for k, v in val_hits.items()}})
            wandb.log({ "epoch": epoch + 1,  "test_mrr": test_mrr,  **{f"test_hits@{k}": v for k, v in test_hits.items()}})
            
            # Update best test performance
            if test_hits[1] + test_mrr > best_test["mrr"] + best_test["hits"].get(1, 0):
                best_test = {"hits": test_hits,"mrr": test_mrr,"epoch": epoch+1 }
                
                # Log best test performance
                wandb.log({ "best_test_epoch": epoch + 1, "best_test_mrr": best_test['mrr'],**{f"best_test_hits@{k}": v for k, v in best_test["hits"].items()}})
                print(f"Updated best test score! ")

                early_stopping = 0
                
                # Save best model checkpoint
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_hits': test_hits,
                    'test_mrr': test_mrr,
                    'self.args': self.args
                }
                
                # Save the best model
                torch.save(best_checkpoint, self.best_model_path)
                print(f"[Updated] Best model saved at {self.best_model_path}")

            else:
                # Increment early stopping counter
                early_stopping += 1
                print(f"Early stopping count: {early_stopping}/{self.args.patience}")
                
            # Check for early stopping condition
            if early_stopping >= self.args.patience:
                print("Early stopping at epoch", epoch)
                break

        wandb.finish()

    def pred_candidate(self):
        # Generate candidate predictions
        print('--------- Candidate Prediction ---------')
        train_preds, val_preds, test_preds = generate_candidate_preds(
            self.model, 
            self.train_loader, 
            self.val_loader, 
            self.test_loader, 
            self.entity_loader,
            self.args.device,
            num_candidate=self.args.num_candidates
        )
        
        candidate_preds = {'train': train_preds, 'val': val_preds, 'test': test_preds }
        
        # # Save candidate predictions
        with open(self.candidate_path, 'w') as f:
            json.dump(candidate_preds, f, indent=4)
            print(f"Saved candidate predictions to {self.candidate_path}")

class RetrievalModel(nn.Module):
    """
    self.Args:
        hidden_dim (int): Dimension of hidden representations
        temperature (float): Temperature parameter for attention softmax
        p (int): Number of top elements to keep in sparse attention
        dropout (float, optional): Dropout probability for MLPs. Defaults to 0.2.
        beta (float, optional): Weighting factor between image and head attention. Defaults to 0.4.
    """
    def __init__(self, hidden_dim, temperature, p, dropout=0.2, beta=0.4):
        super().__init__()
        # Final projection layer for combining features with gating mechanism
        self.final_proj = GatedProj(hidden_dim)

        # MLP to combine relation and tail features
        self.reltail_feature_combine = MLP(2 * hidden_dim, hidden_dim, dropout=dropout)

        # Key/value projections for triple attention mechanism
        self.reltail_key = nn.Linear(hidden_dim, hidden_dim)
        self.reltail_value = nn.Linear(hidden_dim, hidden_dim)
        self.reltail_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Query projections for image and head
        self.img_query = nn.Linear(hidden_dim, hidden_dim)
        self.head_query = nn.Linear(hidden_dim, hidden_dim)

        # Hyperparameters
        self.temperature = temperature  # Controls softmax "sharpness"
        self.p = p  # Number of top tokens to keep in sparse attention
        self.beta = beta  # Weight between image query (beta) and head query (1-beta)


    def compute_attention(self, query1, query2, key, value, mask=None):
        """
        self.Args:
            query1 (torch.Tensor): First query tensor (image)
            query2 (torch.Tensor): Second query tensor (head entity)
            key (torch.Tensor): Key tensor for computing attention scores
            value (torch.Tensor): Value tensor to be aggregated
            mask (torch.Tensor, optional): Attention mask (1 for valid positions, 0 for padding)
            
        Returns:
            torch.Tensor: Context vector from attention mechanism
        """
        # Compute attention scores for both queries (scaled dot-product attention)
        scores1 = torch.matmul(query1, key.transpose(-2, -1)) / math.sqrt(query1.size(-1))
        scores2 = torch.matmul(query2, key.transpose(-2, -1)) / math.sqrt(query2.size(-1))

        # Apply mask if provided
        if mask is not None:
            # Mask out padding tokens
            scores1 = scores1.masked_fill(mask < 0.01, float('-inf'))
            scores2 = scores2.masked_fill(mask < 0.01, float('-inf'))
            # Calculate number of valid positions for determining p
            valid_positions = (mask > 0.01).sum(dim=-1, keepdim=True)
            seq_len = valid_positions.max().item()
        else:
            seq_len = scores1.size(-1)

        # Combine scores from both queries using beta weighting
        scores = scores1 * self.beta + scores2 * (1-self.beta)
        scores = scores / self.temperature
        attn_weights = F.softmax(scores, dim=-1)

        # Apply top-p sparse attention
        p = min(self.p, seq_len)  # Ensure p doesn't exceed sequence length
        _, top_p_indices = torch.topk(attn_weights, p, dim=-1)  # Get indices of top-p scores
        top_p_mask = torch.zeros_like(attn_weights, device=attn_weights.device)
        top_p_mask.scatter_(-1, top_p_indices, 1.0)  # Create binary mask for top-p positions
        attn_weights = attn_weights * top_p_mask  # Apply the mask to keep only top-p weights

        # Aggregate values using attention weights and project
        return self.reltail_proj(torch.matmul(attn_weights, value))


    def forward(self, x_head: torch.Tensor, 
                x_rel: torch.Tensor, 
                x_tail: torch.Tensor,
                x_mask: torch.Tensor, 
                x_img: torch.Tensor) -> torch.Tensor:
        """
        self.Args:
            x_head (torch.Tensor): Head entity embeddings [batch_size, hidden_dim]
            x_rel (torch.Tensor): Relation embeddings [batch_size, hidden_dim]
            x_tail (torch.Tensor): Tail entity embeddings [batch_size, hidden_dim
            x_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            x_img (torch.Tensor): Image embeddings [batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: Final combined representation [batch_size, hidden_dim]
        """
        # Expand mask for attention computation
        attention_mask = x_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Combine relation and tail features
        feature_reltail = torch.cat([x_tail, x_rel], dim=-1)
        x_reltail = x_tail + self.reltail_feature_combine(feature_reltail)
        
        # Generate keys and values for attention
        reltail_keys = self.reltail_key(x_reltail)
        reltail_values = self.reltail_value(x_reltail)
        
        # Generate queries from image and head entity
        img_query = self.img_query(x_img).unsqueeze(1)
        head_query = self.head_query(x_head).unsqueeze(1)
        
        # Compute attention using image and head queries
        x_triple = self.compute_attention(
            query1=img_query, 
            query2=head_query, 
            key=reltail_keys, 
            value=reltail_values, 
            mask=attention_mask
        )
        x_triple = x_triple.squeeze(1)
        
        # Combine all information using gated projection
        return self.final_proj(x_head, x_img, x_triple)

class GatedProj(nn.Module):
    """
    self.Args:
        dim (int): Dimension of input and output features
    """
    def __init__(self, dim):
        super().__init__()
        # Linear projections for each input type
        self.linear_text = nn.Linear(dim, dim)
        self.linear_img = nn.Linear(dim, dim)
        self.linear_triple = nn.Linear(dim, dim)

        # Gating mechanisms for text and image
        self.gate_text = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()  # Ensure values are between 0 and 1
        )
        self.gate_img = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()  # Ensure values are between 0 and 1
        )

    def forward(self, x_text, x_img, x_triple):
        """
        self.Args:
            x_text (torch.Tensor): Text representation
            x_img (torch.Tensor): Image representation
            x_triple (torch.Tensor): Triple representation
            
        Returns:
            torch.Tensor: Combined representation with gated contributions
        """
        # Project each input type
        text_output = self.linear_text(x_text)
        img_output = self.linear_img(x_img)
        triple_output = self.linear_triple(x_triple)

        # Calculate gate values
        text_gate = self.gate_text(x_text)
        img_gate = self.gate_img(x_img)
        
        # Combine outputs with gating
        return triple_output + text_output * text_gate + img_output * img_gate

class MLP(nn.Module):
    """
    self.Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden and output dimension
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        # First linear layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Activation function
        self.act = nn.LeakyReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)  
        # Second linear layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        self.Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Transformed output tensor
        """
        # First layer with activation and dropout
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        # Second layer without activation
        x = self.fc2(x)
        return x

def parse_arguments():
    """
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Multimodal Entity Linking Training and Candidate Generation")
    
    # Data configuration
    parser.add_argument("--base_dir", type=str, default="/workspace/KGMEL", help="Base directory")
    parser.add_argument("--data_dir", type=str, default="/workspace/KGMEL/data", help="Data directory")
    parser.add_argument("--dataset", type=str, default='WikiMEL', help="Dataset to use")
    parser.add_argument("--vlm", type=str, default='gpt-4o-mini-2024-07-18', help="Vision-Language Model")

    # Training configuration
    parser.add_argument("--gpu", type=str, default='0', help="GPU device to use")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Model hyperparameters
    parser.add_argument("--att_temperature", type=float, default=0.1, help="Attention temperature")
    parser.add_argument("--p", type=int, default=5, help="P parameter for attention")
    parser.add_argument("--max_triples_retrieve", type=int, default=1000, help="Maximum number of triples in retrieval, avoiding OOM error")
    parser.add_argument("--beta", type=float, default=0.6, help="Beta parameter")
    parser.add_argument("--cl_temperature", type=float, default=0.1, help="Temperature for loss calculation")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Candidate generation
    parser.add_argument("--num_candidates", type=int, default=16, help="Number of candidates to generate")
    
    # Regularization and loss
    parser.add_argument("--lambda_mm", type=float, default=0.1, help="Multimodal loss weight")
    parser.add_argument("--lambda_ee", type=float, default=0.1, help="Entity embedding loss weight")
    
    # Logging and experiment tracking
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default='KGMEL', help="Weights & Biases project")

    # Miscellaneous
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    return parser.parse_self.args()

if __name__ == "__main__":
    # Parse arguments and run main training and candidate generation script
    self.args = parse_arguments()
    seed_everything(self.args.seed)
    