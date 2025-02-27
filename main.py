import argparse
from module.generate import TripleGenerationModule
from module.retrieve import RetrievalModule
from module.rerank import ReRankingModule


def parse_arguments():
    parser = argparse.ArgumentParser(description="KGMEL Arguments")
    
    # Data configuration
    parser.add_argument("--base_dir", type=str, default="/workspace/KGMEL", help="Base directory")
    parser.add_argument("--data_dir", type=str, default="/workspace/KGMEL/data", help="Data directory")
    parser.add_argument("--dataset", type=str, default='WikiMEL', help="Dataset to use")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default='KGMEL', help="Weights & Biases project")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")

    parser.add_argument("--gpu", type=str, default='0', help="GPU device to use")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    

    # VLM/LLM configuration
    parser.add_argument("--vlm", type=str, default='gpt-4o-mini-2024-07-18', help="VLM for Triple Generation",
                        choices=["gpt-4o-mini-2024-07-18","llava-hf/llava-v1.6-mistral-7b-hf","llava-hf/llava-v1.6-vicuna-13b-hf"])
    parser.add_argument("--llm", type=str, default='gpt-3.5-turbo-0125', help="LLM for Candidate Entity Reranking", 
                        choices=['meta-llama/Llama-2-7b-chat-hf','gpt-3.5-turbo-0125'])

    # Retrieval Training configuration
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--encode_batch_size", type=int, default=1024, help="Encoding batch size")
    parser.add_argument("--max_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")

    parser.add_argument("--lambda_mm", type=float, default=0.1, help="MM Loss weight")
    parser.add_argument("--lambda_ee", type=float, default=0.1, help="EE Loss weight")
    parser.add_argument("--cl_temperature", type=float, default=0.1, help="Temperature for loss calculation")
    
    
    # Retrieval Model configuration
    parser.add_argument("--att_temperature", type=float, default=0.1, help="Temperature for attention")
    parser.add_argument("--beta", type=float, default=0.6, help="Beta parameter")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--p", type=int, default=5, help="P parameter for sparse attention")
    parser.add_argument("--max_triples_retrieve", type=int, default=1000, help="Maximum number of triples in retrieval, avoiding OOM error")

    # Reranking Model configuration
    parser.add_argument("--num_candidates", type=int, default=16, help="Number of candidates to generate")
    parser.add_argument("--n", type=int, default=15, help="N parameter for triple filtering")
    parser.add_argument("--max_triples_rerank", type=int, default=50, help="Maximum number of triples in reranking, avoiding Context Length error")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()


    triple_generation_module = TripleGenerationModule(args)
    retrieval_module = RetrievalModule(args)
    reranking_module = ReRankingModule(args)
    
    # triple_generation_module.generate()
    retrieval_module.retrieve()
    reranking_module.rerank()
