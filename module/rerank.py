from utils.triple_filtering import triple_filtering
from utils.embedding_processor import EmbeddingProcessor
import torch
import wandb
import json
import argparse
from torch_geometric.seed import seed_everything
import pandas as pd
import os 
import numpy as np 
import pandas as pd
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class ReRankingModule:
    def __init__(self, args):
        self.args = args
        
        self.run_name                = f"KGMEL-{self.args.dataset}"
        self.rerank_run_name         = f"KGMEL-{self.args.dataset}-Rerank"
        self.candidate_path          = f"{self.args.base_dir}/checkpoints/{self.run_name}/candidate-{self.args.num_candidates}.json"
        
        self.filtered_tail_path      = f"{self.args.base_dir}/checkpoints/{self.run_name}/filtered-tail_{self.args.num_candidates}-n{self.args.n}.json"
        self.filtered_relation_path  = f"{self.args.base_dir}/checkpoints/{self.run_name}/filtered-relation_{self.args.num_candidates}-n{self.args.n}.json"

        self.json_path               = f"{self.args.base_dir}/output/{self.args.dataset}_{self.args.vlm}_total.json"
        self.rerank_json_path        = self.json_path.replace('.json', f'-{self.args.num_candidates}-n{self.args.n}.json')
        self.split_path              = f"{self.args.data_dir}/dataset/mapping/ids_split_mappings.json"
        

        self.combined_node_mapping_path     = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_node2idx.json"
        self.combined_relation_mapping_path = f"{self.args.base_dir}/embedding/combined/{self.args.dataset}_{self.args.vlm}_relation2idx.json"

        self.k_values = [1,3,5,10,16]

    def rerank(self):
        # Load candidate entities, evaluate retrieval, load data, and prepare rerank data
        self.load_candidate_entity()
        self.evaluate_retrieval()
        self.load_data()
        self.load_mapping()

        # Perform triple filtering and prepare rerank data
        self.triple_filtering()
        self.preprare_rerank_data()

        # Zero-shot Reranking 
        if 'gpt' in self.args.llm.lower() : 
            return self.gpt_rerank()
        elif 'llama' in  self.args.llm.lower():
            return self.llama_rerank()
        else :
            raise ValueError(f"Invalid VLM model: {self.args.vlm}")


    def load_candidate_entity(self):
        print('--------- Loading Candidate Entities ---------')
        with open(self.candidate_path, 'r') as f:
            self.preds = json.load(f)
            self.test_preds = self.preds['test']
        print(f"Loaded Candidate entities from {self.candidate_path}")
            
    def evaluate_retrieval(self):
        print('--------- Evaluating Retrieval ---------')
        retrieval_rate ={'train':0,'val':0,'test':0}
        for split in ['train','val','test']:
            cnt = sum( a in c for a, c in zip(self.preds[split]['answer'], self.preds[split]['candidate']))
            retrieval_rate[split] = 100 * cnt / len(self.preds[split]['mention_key'])
        print(f"Retrieval@{self.args.num_candidates} - Train: {retrieval_rate['train']:.2f} Val: {retrieval_rate['val']:.2f} Test: {retrieval_rate['test']:.2f}")
    
    def load_data(self):
        print('--------- Loading Data ---------')
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)
        with open(self.split_path, 'r') as f:
            split2id = json.load(f)
        test_ids =  split2id[self.args.dataset]['test']
        self.test_data =  [item for item in json_data if item['id'] in test_ids]
        print(f"Loaded Test Data for {self.args.dataset} - {self.args.vlm}")

    def triple_filtering(self):
        print('--------- Triple Filtering ---------')
        if os.path.exists(self.filtered_tail_path) and os.path.exists(self.filtered_relation_path):
            print(f"Filtered triples already exist at {self.filtered_tail_path} and {self.filtered_relation_path}")
            with open(self.filtered_tail_path, 'r') as f:
                self.filtered_test_tail = json.load(f)
            with open(self.filtered_relation_path, 'r') as f:
                self.filtered_test_rel = json.load(f)
            return 
        data = EmbeddingProcessor(self.args)
        self.filtered_test_rel, self.filtered_test_tail = triple_filtering(data, self.test_preds, self.args)
        with open(self.filtered_tail_path, 'w') as f:
            json.dump(self.filtered_test_tail, f, indent=4)
        with open(self.filtered_relation_path, 'w') as f:
            json.dump(self.filtered_test_rel, f, indent=4)
        print(f"Filtered triples saved to {self.filtered_tail_path} and {self.filtered_relation_path}")
    
    def load_mapping(self):
        print('--------- Loading Mapping ---------')
        # for Entity
        qid_label_path = f"{self.args.data_dir}/KB/QID2Label.tsv"
        pid_label_path = f"{self.args.data_dir}/KB/PID2Label.tsv" 
        qid = pd.read_csv(qid_label_path, sep='\t', names=['qid', 'label', 'desc'])
        pid = pd.read_csv(pid_label_path, sep='\t', names=['pid', 'label', 'desc'])
        self.qid2desc = dict(zip(qid['qid'], qid['desc']))
        self.qid2label = dict(zip(qid['qid'], qid['label']))
        self.pid2label = dict(zip(pid['pid'], pid['label']))
        

        # for Mention
        self.mention_key2cand, self.mention_key2rank = {}, {}
        for mention_key, candidate, rank in zip(self.test_preds['mention_key'], self.test_preds['candidate'], self.test_preds['rank']):
            self.mention_key2cand[mention_key] = candidate
            self.mention_key2rank[mention_key] = rank

        # for Triple
        with open(self.combined_relation_mapping_path, 'r') as f:
            rel2idx = json.load(f)
            self.idx2rel = {v:k for k,v in rel2idx.items()}
        with open(self.combined_node_mapping_path, 'r') as f:
            node2idx = json.load(f)
            self.idx2node = {v:k for k,v in node2idx.items()}
        print (f"Loaded Mapping for {self.args.dataset} - {self.args.vlm}")

    def process_candidate(self, candidate, item, a):
        cand_data = []
        for i, c in enumerate(candidate):
            cand_data.append({'qid': c, 'label': self.qid2label[c], 'desc': self.qid2desc[c]})
            rel_idx = self.filtered_test_rel[f"{item['id']}-{a}-{c}"]
            tail_idx = self.filtered_test_tail[f"{item['id']}-{a}-{c}"]
            triple = []
            for z,t in zip(rel_idx, tail_idx):
                rel_id, tail_id = self.idx2rel[z], self.idx2node[t]
                rel_label, tail_label = self.pid2label[rel_id], self.qid2label[tail_id]
                triple.append(f"{rel_label} {tail_label}")
            cand_data[i]['triple'] = triple
        return cand_data

    def preprare_rerank_data(self):
        print('--------- Preparing Rerank Data ---------')
        self.rerank_test_data =[]
        for test_item in self.test_data: 
            for a,m,e in zip(test_item['answer'], test_item['mention'], test_item['entity']):
                item = ({'id': test_item['id'], 
                        'sentence': test_item['sentence'], 
                        'mention': m, 'answer': a, 'label': e,
                        'mention-desc': test_item['desc'].get(m, None), 
                        'mention-triple': test_item['triple'].get(m, None), 
                        'retrieve-hit' : None, 
                        'retrieve-rank': self.mention_key2rank[f"{test_item['id']}-{a}"],
                        'candidate': []})
                item['mention-triple'] = [f"{t[1]} {t[2]}" for t in item['mention-triple']] if item['mention-triple'] else None

                candidate = self.mention_key2cand[f"{item['id']}-{a}"]
                if a not in candidate:
                    item['hit'] = 0 
                    self.rerank_test_data.append(item)
                    continue
                item['candidate'] = self.process_candidate(candidate, item, a)

                self.rerank_test_data.append(item)

        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f, indent=2)
        print(f"Prepared Rerank Data for {self.args.dataset} at {self.rerank_json_path}")

    def format_prompt_gpt(self,item):
        candidate = [f"{t['label']} ({t['qid']}):{t['desc']}  \nTriple: {'; '.join(t['triple'])}" for t in item['candidate']][::-1]
        mention_triple = '\n'.join(item['mention-triple'])
        candidate = '\n'.join(candidate)
        return f"""Given the context below, please identify the most corresponding entity from the list of candidates.

Context: {item['sentence']}

Candidate Entities:
{ candidate }

Context: {item['sentence']}

Target Entity: "{item['mention']}": {item['mention-desc']}
Triple: {mention_triple}

First, read the context and the target entity. Then, review the candidate entities and their descriptions.
From the candidate entities, select the supporting triples that align with the context and the target entity. (Note that triples may contain inconsistent information.)
Based on the selected supporting triples, identify the most relevant entity that best matches the given sentence context."""

    def verify_response(self, response, label, answer):
        # for LLaMA response often contains multiple QIDs we use the first QID
        qid = re.search(r'Q\d+', response)
        if qid :
            if qid.group(0) == answer :
                return 1
        # Check if the label appears in the response (case insensitive)
        if  label.lower() in response.lower():
            return 1
        return 0 
    
    def evaluate(self, ranks):
        # Print Hit@k metrics for each k value in self.k_values
        for k in self.k_values:
            hits = sum([1 for r in ranks if r <= k])
            hit_rate = 100 * hits / len(ranks)
            print(f"H@{k}: {hit_rate:.2f}", end=" ")
        
        # Calculate and print Mean Reciprocal Rank (MRR)
        mrr = 100 * np.mean([1/r for r in ranks])
        print(f"MRR: {mrr:.2f}")
        
        # Prepare results dictionary
        result = {"MRR": mrr}
        
        # Add Hit@k metrics to results dictionary
        result.update({
            f"H@{k}": 100 * sum([1 for r in ranks if r <= k]) / len(ranks) 
            for k in self.k_values
        })
        
        return result

    def gpt_rerank(self):
        print('--------- Reranking with GPT ---------')
        # Initialize wandb tracking
        wandb.init(
            project=self.args.wandb_project, 
            name=self.rerank_run_name
        )
        wandb.config.update(self.args)
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Initialize rank tracking lists
        ranks, rerank_ranks = [], []
        
        # Process each test item
        for i, item in enumerate(self.rerank_test_data):
            # --- Initialize reranking fields ---
            self.rerank_test_data[i]['rerank-response'] = ''
            self.rerank_test_data[i]['rerank-hit'] = 0
            self.rerank_test_data[i]['rerank-rank'] = item['retrieve-rank']
            
            # --- Skip items where answer is not in candidates ---
            if item['retrieve-hit'] == 0:
                ranks.append(self.rerank_test_data[i]['retrieve-rank'])
                rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])
                print(f"Answer NOT in Candidate")
                continue
                
            # --- Get LLM response ---
            try:
                # Generate completion using OpenAI API
                completion = client.chat.completions.create(
                    model=self.args.llm,
                    messages=[
                        {
                            "role": "developer", 
                            "content": "You are a helpful assistant that helps users find the most relevant entity from a list of candidates."
                        },
                        {
                            "role": "user",
                            "content": self.format_prompt_gpt(item)
                        }
                    ]
                )
                
                # Store response
                response = completion.choices[0].message.content
                self.rerank_test_data[i]['rerank-response'] = response
                
                # Print debug information
                print('='*50)
                print(f"[Answer] {item['answer']}")
                print(f"[Label] {item['label']}")
                print(f"[Response] {response}")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            # --- Evaluate response ---
            # Check if response matches expected answer or label
            self.rerank_test_data[i]['rerank-hit'] = self.verify_response(
                self.rerank_test_data[i]['rerank-response'], 
                item['label'], 
                item['answer']
            )
            
            # Set rank position based on hit status
            self.rerank_test_data[i]['rerank-rank'] = 1 if item['rerank-hit'] == 1 else item['retrieve-rank']+1
            
            # Add to rank tracking lists
            ranks.append(item['retrieve-rank'])
            rerank_ranks.append(item['rerank-rank'])
            
            # --- Log and save metrics ---
            # Calculate and log current metrics
            print(f"Retrieval Result:", end=" ")
            retrieve_eval = self.evaluate(ranks)
            print(f"Reranking Result:", end=" ")
            rerank_eval = self.evaluate(rerank_ranks)
            
            # Verification check
            assert len(ranks) == len(rerank_ranks)
            
            # Log to wandb
            wandb.log({'rerank': rerank_eval, 'retrieve': retrieve_eval})
            
            # Save intermediate results at regular intervals
            if i % 50 == 0:
                with open(self.rerank_json_path, 'w') as f:
                    json.dump(self.rerank_test_data, f)
                print(f"Saved reranking results to {self.rerank_json_path}")

        
        # --- Save final results ---
        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f)
        print(f"Saved reranking results to {self.rerank_json_path}")
        
        # Calculate final evaluation metrics
        rerank_eval = self.evaluate(rerank_ranks)
        retrieve_eval = self.evaluate(ranks)
        
        # Log final results to wandb
        wandb.log({'final-rerank': rerank_eval, 'final-retrieve': retrieve_eval})
        
        return rerank_eval, retrieve_eval

    def format_prompt_llama(self, item):
        candidate = [f"{t['label']} ({t['qid']}):{t['desc']}  \nTriple: {'; '.join(t['triple'])}" for t in item['candidate']][::-1]
        mention_triple = '\n'.join(item['mention-triple'])
        candidate = '\n'.join(candidate)
        return  f"""<s>[INST] <<SYS>>
You are a helpful assistant that helps users find the most relevant entity from a list of candidates.
<</SYS>>

Given the context below, please identify the most corresponding entity from the list of candidates.
Context: {item['sentence']}
Candidate Entities:
{candidate}
Context: {item['sentence']}
Target Entity: "{item['mention']}": {item['mention-desc']}
Triple: {mention_triple}

Instructions:
First, read the context and the target entity. Then, review the candidate entities and their descriptions.
From the candidate entities, select the supporting triples that align with the context and the target entity. (Note that triples may contain inconsistent information.)
Based on the selected supporting triples, identify the MOST relevant entity that best matches the given sentence context.
Please provide ONLY ONE entity with its QID which is the most relevant to the given context and target entity.[/INST]"""

    def llama_rerank(self):
        print('--------- Reranking with LLaMA ---------')
        # Initialize wandb tracking
        wandb.init(
            project=self.args.wandb_project, 
            name=self.rerank_run_name
        )
        wandb.config.update(self.args)
        
        login(token=os.environ.get("HUGGINGFACE_TOKEN"))

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.llm)
        model = AutoModelForCausalLM.from_pretrained(self.args.llm,
                                                    torch_dtype=torch.float16,
                                                    # cache_dir=self.args.huggingface_cache_dir,
                                                    use_auth_token=True,
                                                    device_map="auto").to(self.args.device)
    
        # Initialize rank tracking lists
        ranks, rerank_ranks = [], []
        
        # Process each test item
        for i, item in enumerate(self.rerank_test_data):
            # --- Initialize reranking fields ---
            self.rerank_test_data[i]['rerank-response'] = ''
            self.rerank_test_data[i]['rerank-hit'] = 0
            self.rerank_test_data[i]['rerank-rank'] = item['retrieve-rank']
            
            # --- Skip items where answer is not in candidates ---
            if item['retrieve-hit'] == 0:
                ranks.append(self.rerank_test_data[i]['retrieve-rank'])
                rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])
                print(f"Answer NOT in Candidate")
                continue
                
            # --- Get LLM response ---
            try:
                # Generate completion using OpenAI API
                inputs = tokenizer(self.format_prompt_llama(item), return_tensors="pt").to("cuda")
                output = model.generate(**inputs, max_new_tokens=512)
                response = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
                
                
                # Store response
                self.rerank_test_data[i]['rerank-response'] = response
       
                # Print debug information
                print('='*50)
                print(f"[Answer] {item['answer']}")
                print(f"[Label] {item['label']}")
                print(f"[Response] {response}")
                    
            except Exception as e:
                print(f"Error: {e}")
            
            # --- Evaluate response ---
            # Check if response matches expected answer or label
            self.rerank_test_data[i]['rerank-hit'] = self.verify_response(
                self.rerank_test_data[i]['rerank-response'], 
                item['label'], 
                item['answer']
            )
            
            # Set rank position based on hit status
            self.rerank_test_data[i]['rerank-rank'] = 1 if item['rerank-hit'] == 1 else item['retrieve-rank']+1
            
            # Add to rank tracking lists
            ranks.append(item['retrieve-rank'])
            rerank_ranks.append(item['rerank-rank'])
            
            # --- Log and save metrics ---
            # Calculate and log current metrics
            print(f"Retrieval Result:", end=" ")
            retrieve_eval = self.evaluate(ranks)
            print(f"Reranking Result:", end=" ")
            rerank_eval = self.evaluate(rerank_ranks)
            
            # Verification check
            assert len(ranks) == len(rerank_ranks)
            
            # Log to wandb
            wandb.log({'rerank': rerank_eval, 'retrieve': retrieve_eval})
            
            # Save intermediate results at regular intervals
            if i % 50 == 0:
                with open(self.rerank_json_path, 'w') as f:
                    json.dump(self.rerank_test_data, f)
                print(f"Saved reranking results to {self.rerank_json_path}")

        
        # --- Save final results ---
        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f)
        print(f"Saved reranking results to {self.rerank_json_path}")
        
        # Calculate final evaluation metrics
        rerank_eval = self.evaluate(rerank_ranks)
        retrieve_eval = self.evaluate(ranks)
        
        # Log final results to wandb
        wandb.log({'final-rerank': rerank_eval, 'final-retrieve': retrieve_eval})
        wandb.finish()  
        return rerank_eval, retrieve_eval
    
    
