import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm

def get_dataloaders(data, args):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data: Data with embeddings and graph structures
        args: Arguments containing configuration parameters
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    datasets = {}
    loaders = {}
    
    # Create datasets for each split
    for split in ['train', 'val', 'test']:
        # Create the base dataset
        datasets[split] = KGDataset(
            M_node=data.node_embeddings,
            M_rel=data.rel_embeddings,
            M_img=data.img_embeddings,
            G_entity_head=data.G_entity_head,
            G_entity_tail=data.G_entity_tail,
            G_entity_relation=data.G_entity_relation,
            G_entity_img=data.G_entity_img,
            G_mention_head=data.G_mention_head[split],
            G_mention_tail=data.G_mention_tail[split],
            G_mention_relation=data.G_mention_relation[split],
            G_mention_img=data.G_mention_img[split],
            max_triples_retrieve=args.max_triples_retrieve,
        )
        
        # Create corresponding dataloader
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == 'train'),  # Only shuffle training data
            collate_fn=collate_kg,
            num_workers=args.num_workers
        )
    
    return loaders['train'], loaders['val'], loaders['test']


def get_entity_dataloaders(data, args):
    """
    Create a DataLoader for entity embeddings.
    
    Args:
        data: Data containing embeddings and graph structures
        args: Arguments containing batch size, max_triples_retrieve, and other configuration
        
    Returns:
        DataLoader: DataLoader for entity with appropriate batching and collation
    """
    # Create entity evaluation dataset
    entity_dataset = EntityKGDataset(
        M_node=data.node_embeddings,
        M_rel=data.rel_embeddings,
        M_img=data.img_embeddings,
        G_entity_head=data.G_entity_head,
        G_entity_tail=data.G_entity_tail,
        G_entity_relation=data.G_entity_relation,
        G_entity_img=data.G_entity_img,
        max_triples_retrieve=args.max_triples_retrieve
    )
    
    # Create DataLoader with appropriate collation function
    entity_loader = DataLoader(
        entity_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling for evaluation
        collate_fn=collate_entity,
        num_workers=args.num_workers
    )
    return entity_loader


class KGDataset(Dataset):
    def __init__(self, M_node, M_rel, M_img, 
                G_entity_head, G_entity_relation, G_entity_tail, G_entity_img,
                G_mention_head, G_mention_relation, G_mention_tail, G_mention_img, 
                max_triples_retrieve=1000):
        # embeddings (all 512 dim)
        self.M_node = torch.tensor(M_node)  # [N, 512]
        self.M_rel = torch.tensor(M_rel)    # [N, 512]
        self.M_img = torch.tensor(M_img)    # [N, 512]

        # Embedding indices for mentions
        self.G_mention_head = G_mention_head
        self.G_mention_tail = G_mention_tail 
        self.G_mention_relation = G_mention_relation
        self.G_mention_img = G_mention_img

        # Embedding indices for entities
        self.G_entity_head = G_entity_head
        self.G_entity_tail = G_entity_tail
        self.G_entity_relation = G_entity_relation
        self.G_entity_img = G_entity_img

        self.entity_qids = list(self.G_entity_head.keys())
        self.mention_keys = list(self.G_mention_head.keys())
        self.max_triples_retrieve = max_triples_retrieve

        # Dimensions for embeddings
        self.embed_dim = self.M_node.size(1)  

        # Zero embeddings
        self.zero_emb = torch.zeros(1, self.embed_dim)

        self.num_entities = len(self.entity_qids)

    def __len__(self):
        return len(self.mention_keys)

    def get_node_embeddings(self, indices):
        """
        Retrieve node embeddings for the given indices.
        
        Args:
            indices (List[int]): List of node indices to retrieve
            
        Returns:
            torch.Tensor: Node embeddings, or zero embedding if indices is empty
        """
        if len(indices) == 0:
            return self.zero_emb
        
        node_emb = self.M_node[indices]
        
        # Sample if exceeding max_triples_retrieve
        if node_emb.size(0) > self.max_triples_retrieve:
            idx = torch.randperm(node_emb.size(0))[:self.max_triples_retrieve]
            node_emb = node_emb[idx]
        
        return node_emb

    def get_relation_embeddings(self, indices):
        """
        Retrieve relation embeddings for the given indices.
        
        Args:
            indices (List[int]): List of relation indices to retrieve
            
        Returns:
            torch.Tensor: Relation embeddings, or zero embedding if indices is empty/None
        """
        if indices is None or len(indices) == 0:
            return self.zero_emb
        
        rel_emb = self.M_rel[indices]
        
        # Sample if exceeding max_triples_retrieve
        if rel_emb.size(0) > self.max_triples_retrieve:
            idx = torch.randperm(rel_emb.size(0))[:self.max_triples_retrieve]
            rel_emb = rel_emb[idx]
        
        return rel_emb

    def get_image_embedding(self, index):
        """
        Retrieve image embedding for the given index.
        
        Args:
            index (int or None): Image index to retrieve
            
        Returns:
            torch.Tensor: Image embedding, or zero embedding if index is None
        """
        if index is None:
            return self.zero_emb.squeeze(0)
        return self.M_img[index]

    def get_entity_components(self, qid):
        """
        Retrieve all embeddings for a given entity.
        
        Args:
            qid (str): Entity identifier
            
        Returns:
            tuple: (head_embedding, relation_embeddings, tail_embeddings, image_embedding)
        """
        head_emb = self.M_node[self.G_entity_head[qid]]
        image_emb = self.get_image_embedding(self.G_entity_img[qid])
        
        tail_indices = self.G_entity_tail.get(qid, [])
        relation_indices = self.G_entity_relation.get(qid, [])
        
        tail_emb = self.get_node_embeddings(tail_indices)
        relation_emb = self.get_relation_embeddings(relation_indices)
        
        return head_emb, relation_emb, tail_emb, image_emb

    def __getitem__(self, idx):
        mention_key = self.mention_keys[idx]
        positive_qid = mention_key.split('-')[-1]

        # Get mention embeddings
        mention_head_emb = self.M_node[self.G_mention_head[mention_key]]

        mention_tail_indices = self.G_mention_tail[mention_key]
        mention_relation_indices = self.G_mention_relation.get(mention_key, None)
        
        mention_tail_emb = self.get_node_embeddings(mention_tail_indices)
        mention_relation_emb = self.get_relation_embeddings(mention_relation_indices)
        mention_img_emb = self.get_image_embedding(self.G_mention_img[mention_key])

        # Get embeddings for positive entities
        pos_head_emb, pos_relation_emb, pos_tail_emb, pos_img_emb = self.get_entity_components(positive_qid)

        return {
            'mention_head_emb': mention_head_emb,
            'mention_tail_emb': mention_tail_emb,
            'mention_relation_emb': mention_relation_emb,
            'mention_img_emb': mention_img_emb,

            'positive_head_emb': pos_head_emb,
            'positive_relation_emb': pos_relation_emb,
            'positive_tail_emb': pos_tail_emb,
            'positive_img_emb': pos_img_emb,

            'answer': positive_qid,
            'mention_key': mention_key
        }


class EntityKGDataset(Dataset):
    def __init__(
        self,
        M_node, M_rel, M_img,
        G_entity_head, G_entity_tail, G_entity_relation, G_entity_img,
        max_triples_retrieve=1000, 
    ):
        # Convert numpy arrays to tensors if needed
        self.M_node = torch.tensor(M_node) 
        self.M_rel = torch.tensor(M_rel) 
        
        # Keep pooled image embeddings
        self.M_img = torch.tensor(M_img) 

        # Graph structures
        self.G_entity_head = G_entity_head
        self.G_entity_tail = G_entity_tail
        self.G_entity_relation = G_entity_relation
        self.G_entity_img = G_entity_img

        self.entity_qids = list(self.G_entity_head.keys())
        self.max_triples_retrieve = max_triples_retrieve

        # Set dimensions
        self.embed_dim = self.M_node.size(1)

        # Zero embeddings
        self.zero_emb = torch.zeros(1, self.embed_dim)

    def __len__(self):
        return len(self.entity_qids)

    def get_node_embeddings(self, indices):
        """
        Retrieve node embeddings for the given indices.
        
        Args:
            indices (List[int]): List of node indices to retrieve
            
        Returns:
            torch.Tensor: Node embeddings, or zero embedding if indices is empty
        """
        if len(indices) == 0:
            return self.zero_emb
        
        node_emb = self.M_node[indices].float()
        
        # Sample if exceeding max_triples_retrieve
        if node_emb.size(0) > self.max_triples_retrieve:
            idx = torch.randperm(node_emb.size(0))[:self.max_triples_retrieve]
            node_emb = node_emb[idx]
        
        return node_emb

    def get_relation_embeddings(self, indices):
        """
        Retrieve relation embeddings for the given indices.
        
        Args:
            indices (List[int]): List of relation indices to retrieve
            
        Returns:
            torch.Tensor: Relation embeddings, or zero embedding if indices is empty/None
        """
        if indices is None or len(indices) == 0:
            return self.zero_emb
        
        rel_emb = self.M_rel[indices].float()
        
        # Sample if exceeding max_triples_retrieve
        if rel_emb.size(0) > self.max_triples_retrieve:
            idx = torch.randperm(rel_emb.size(0))[:self.max_triples_retrieve]
            rel_emb = rel_emb[idx]
        
        return rel_emb

    def get_image_embedding(self, index):
        """
        Retrieve image embedding for the given index.
        
        Args:
            index (int or None): Image index to retrieve
            
        Returns:
            torch.Tensor: Image embedding, or zero embedding if index is None
        """
        if index is None:
            return self.zero_emb.squeeze(0)
        return self.M_img[index].float()
    
    def __getitem__(self, idx: int):
        qid = self.entity_qids[idx]

        # Get head embeddings
        head_indices = self.G_entity_head[qid]
        head_emb = self.M_node[head_indices].float()

        # Get image embeddings
        img_emb = self.get_image_embedding(self.G_entity_img[qid])

        # Get tail and relation indices
        tail_indices = self.G_entity_tail.get(qid, [])
        relation_indices = self.G_entity_relation.get(qid, [])
        
        # Get tail and relation embeddings
        tail_emb = self.get_node_embeddings(tail_indices)
        relation_emb = self.get_relation_embeddings(relation_indices)

        return {
            'head_emb': head_emb,
            'tail_emb': tail_emb,
            'relation_emb': relation_emb,
            'img_emb': img_emb,
            'qid': qid
        }


def collate_kg(batch):
    """
    Collate function for knowledge graph batches that handles variable-length sequences.
    
    Args:
        batch: List of dictionaries containing individual samples
        
    Returns:
        dict: Batched tensors with padded sequences and masks
    """
    # Stack fixed-size embeddings
    mention_head_embs = torch.stack([b['mention_head_emb'] for b in batch])
    mention_img_embs = torch.stack([b['mention_img_emb'] for b in batch])
    
    # Process mention tail and relation embeddings
    mention_tail_embs = [b['mention_tail_emb'] for b in batch]
    mention_relation_embs = [b['mention_relation_emb'] for b in batch]
    
    mention_tail_emb_pad = pad_sequence(mention_tail_embs, batch_first=True)
    mention_relation_emb_pad = pad_sequence(mention_relation_embs, batch_first=True)
    
    # Create single mask for mention sequences
    mention_mask = make_mask(mention_tail_emb_pad)  # [batch_size, max_mention_length]

    # Process positive embeddings
    positive_head_embs = torch.stack([b['positive_head_emb'] for b in batch])
    positive_img_embs = torch.stack([b['positive_img_emb'] for b in batch])
    
    positive_tail_embs = [b['positive_tail_emb'] for b in batch]
    positive_relation_embs = [b['positive_relation_emb'] for b in batch]
    
    positive_tail_emb_pad = pad_sequence(positive_tail_embs, batch_first=True)
    positive_relation_emb_pad = pad_sequence(positive_relation_embs, batch_first=True)
    
    # Create single mask for positive sequences
    positive_mask = make_mask(positive_tail_emb_pad)

    return {
        # Mention embeddings
        'mention_head_emb': mention_head_embs,
        'mention_img_emb': mention_img_embs,
        'mention_tail_emb': mention_tail_emb_pad,
        'mention_relation_emb': mention_relation_emb_pad,
        'mention_mask': mention_mask,  # Single mask for both tail and relation

        # Positive entity embeddings
        'positive_head_emb': positive_head_embs,
        'positive_img_emb': positive_img_embs,
        'positive_tail_emb': positive_tail_emb_pad,
        'positive_relation_emb': positive_relation_emb_pad,
        'positive_mask': positive_mask,  # Single mask for both tail and relation
        
        # Other information
        'answer': [b['answer'] for b in batch],
        'mention_key': [b['mention_key'] for b in batch]
    }

def collate_entity(batch):
    """
    Collate function for entity batches that handles variable-length sequences.
    
    Args:
        batch: List of dictionaries containing individual samples
        
    Returns:
        dict: Batched tensors with padded sequences and masks
    """
    # Handle tail and relation embeddings padding
    tail_emb_pad = pad_sequence([b['tail_emb'] for b in batch], batch_first=True)
    relation_emb_pad = pad_sequence([b['relation_emb'] for b in batch], batch_first=True)
    
    # Create single mask for tail sequences (also used for relations)
    entity_mask = make_mask(tail_emb_pad)

    return {
        'head_emb': torch.stack([b["head_emb"] for b in batch]),
        'img_emb': torch.stack([b["img_emb"] for b in batch]),
        'tail_emb': tail_emb_pad,
        'relation_emb': relation_emb_pad,
        'entity_mask': entity_mask,  # Single mask for both tail and relation
        'qid': [b['qid'] for b in batch]
    }


def make_mask(padded_tensor):
    """
    Generate a mask for padded sequences since all triple lengths for each entity are different.
    
    Args:
        padded_tensor (torch.Tensor): Padded tensor of shape [batch_size, max_len, feature_dim]
        
    Returns:
        torch.Tensor: Binary mask of shape [batch_size, max_len] where 1 indicates valid positions
    """
    # padded_tensor: [batch_size, max_len, feature_dim]
    sequences = padded_tensor.unbind(0)  # list of [max_len, feature_dim]
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    max_len = padded_tensor.size(1)
    mask = (torch.arange(max_len)[None, :] < lengths[:, None]).float() 
    return mask
