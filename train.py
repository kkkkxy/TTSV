import argparse
import os
import random
import time
from pathlib import Path
import sys

# Add the evaluation directory to the system path to import its modules
sys.path.append('Qwen2.5-Eval/evaluation')

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Import data processing functions from the evaluation script's directory
from data_loader import load_data
from parser import parse_question
from utils import construct_prompt


os.environ.setdefault("NCCL_TIMEOUT", "2700")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "2700")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen2.5-Math-7B', help='Model name from Hugging Face Hub')
    parser.add_argument('--model_path', type=str, default=None, help='Local model path (overrides model_name)')
    parser.add_argument('--train_dataset_name', type=str, default='gsm8k', help='Training dataset name from Hugging Face Hub')
    parser.add_argument('--data_dir', type=str, default='Qwen2.5-Eval/evaluation/data', help='Directory for the dataset')
    parser.add_argument('--prompt_type', type=str, default='qwen25-math-cot', help='Prompt type, consistent with evaluation')
    parser.add_argument('--num_shots', type=int, default=0, help='Number of few-shot examples')
    parser.add_argument('--save_root', type=str, default='checkpoints', help='Checkpoint save root directory')
    parser.add_argument('--prefix_length', type=int, default=20, help='Length of the trainable prefix')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for entropy calculation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the prefix')
    parser.add_argument('--log_steps', type=int, default=1, help='Logging step interval within an epoch')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--sample_temp', type=float, default=0.5, help='Generation temperature parameter')
    parser.add_argument('--run_name', type=str, default=None, help='Experiment run name for TensorBoard')
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    return parser.parse_args()

class PrefixTuningModel(nn.Module):
    def __init__(self, base_model, prefix_length=20):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.embedding_size = self.base_model.config.hidden_size
        self.config = self.base_model.config

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create a trainable prefix
        self.prefix_embeddings = nn.Embedding(self.prefix_length, self.embedding_size)

    def _prepare_inputs(self, input_ids, attention_mask=None):
        # Get word embeddings from the base model's embedding layer
        word_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Get prefix embeddings
        prefix_tokens = torch.arange(self.prefix_length, device=input_ids.device)
        prefix_embeds = self.prefix_embeddings(prefix_tokens).unsqueeze(0).expand(input_ids.size(0), -1, -1)

        # Ensure prefix embeddings match the dtype of the base model's embeddings
        prefix_embeds = prefix_embeds.to(word_embeddings.dtype)

        # Concatenate prefix and word embeddings
        inputs_embeds = torch.cat([prefix_embeds, word_embeddings], dim=1)
        
        # Create a corresponding attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
        return inputs_embeds, attention_mask

    def forward(self, input_ids, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask = self._prepare_inputs(input_ids, attention_mask)
        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        inputs_embeds, attention_mask = self._prepare_inputs(input_ids, attention_mask)
        
        # The base model's generate method can handle inputs_embeds directly
        return self.base_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

class FTDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def custom_collate(batch):
    return {"input": [item["input"] for item in batch]}

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run_name = args.run_name or f"{args.model_name.replace('/', '_')}-prefix-{args.prefix_length}-{time.strftime('%Y%m%d-%H%M%S')}"
    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.save_root,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # For older accelerate versions, combine project and run name into a single string
    full_run_name = f"tensorboard_logs/{run_name}"
    accelerator.init_trackers(project_name=full_run_name, config=vars(args))

    print = accelerator.print
    
    model_path = args.model_path or args.model_name
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False
    
    base_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model = PrefixTuningModel(
        base_model,
        prefix_length=args.prefix_length,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    print(f"Loading and preparing dataset '{args.train_dataset_name}' using evaluation script's logic...")
    # The evaluation script's functions expect an `args` object with specific attributes.
    # We create a mock 'args' object to pass to them, ensuring compatibility.
    eval_args = argparse.Namespace(**vars(args))
    eval_args.split = "test"  # We are optimizing on the test set
    eval_args.num_test_sample = -1  # Use all samples
    eval_args.adapt_few_shot = False # Add missing attribute expected by eval script

    examples = load_data(args.train_dataset_name, eval_args.split, args.data_dir)

    train_prompts = []
    for example in examples:
        example['question'] = parse_question(example, args.train_dataset_name)
        if example["question"] == "":
            continue
        # Construct the prompt exactly as it would be during evaluation
        full_prompt = construct_prompt(example, args.train_dataset_name, eval_args)
        train_prompts.append(full_prompt)
    
    train_data = [{"input": p} for p in train_prompts]
    train_loader = DataLoader(FTDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, eps=1e-5) 

    scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.num_epochs * accelerator.num_processes
        )
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    
    global_step = 0
    for epoch in range(args.num_epochs):
        print(f"\n--- Starting Epoch {epoch+1}/{args.num_epochs} ---")
        model.train()
        
        total_epoch_entropy = 0.0
        total_epoch_tokens = 0
        
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    enc = tokenizer(batch["input"], return_tensors="pt", padding="longest", truncation=True, max_length=2048).to(accelerator.device)
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    gen_ids = unwrapped_model.generate(**enc,
                                                    max_new_tokens=256,
                                                    do_sample=True,
                                                    top_p=0.95,
                                                    temperature=args.sample_temp,
                                                    synced_gpus=True,
                                                    repetition_penalty=1.15,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    use_cache=True) # Can use cache here
                    
                seq_ids = torch.cat([enc.input_ids, gen_ids], dim=1)
                seq_attention_mask = (seq_ids != tokenizer.pad_token_id)
                
                logits = model(input_ids=seq_ids, attention_mask=seq_attention_mask).logits

                response_logits_start_idx = args.prefix_length + enc.input_ids.shape[1]
                logits_for_loss = logits[:, response_logits_start_idx - 1: -1, :]
                
                probs = torch.softmax(logits_for_loss, dim=-1)
                log_probs = torch.log_softmax(logits_for_loss, dim=-1)
                entropy_per_token = -torch.sum(probs * log_probs, dim=-1)

                response_pad_mask = gen_ids.ne(unwrapped_model.config.pad_token_id or unwrapped_model.config.eos_token_id)
                masked_entropy = entropy_per_token * response_pad_mask

                batch_loss = masked_entropy.sum()
                batch_tokens = response_pad_mask.sum()

                if batch_tokens > 0:
                    batch_mean_loss = batch_loss / batch_tokens
                else:
                    batch_mean_loss = torch.tensor(0.0, device=accelerator.device)

                accelerator.backward(batch_mean_loss)
                optimizer.step()
                optimizer.zero_grad()

                total_epoch_entropy += batch_loss.detach()
                total_epoch_tokens += batch_tokens.detach()
                
                if accelerator.sync_gradients:
                    global_step += 1
                    if accelerator.is_main_process and global_step % args.log_steps == 0:
                        if batch_tokens > 0:
                            current_loss = batch_mean_loss
                            print(f"Epoch {epoch+1} | Update Step {global_step} | Batch Loss={current_loss.item():.6f}")
                            accelerator.log({"update_loss": current_loss.item()}, step=global_step)

        # --- End of Epoch ---
        if total_epoch_tokens == 0:
            print("No tokens were generated in this epoch. Skipping optimization.")
            scheduler.step()
            continue
            
        epoch_loss = total_epoch_entropy / total_epoch_tokens

        if accelerator.is_main_process:
            print(f"--- Epoch {epoch+1} Summary ---")
            print(f"Total Loss = {epoch_loss.item():.6f}")
            # Log epoch-level loss
            logs = {
                "epoch_loss": epoch_loss.item(),
                "learning_rate": scheduler.get_last_lr()[0] 
            }
            accelerator.log(logs, step=epoch+1)
            
            # Save the prefix embeddings
            ckpt_dir = Path(args.save_root) / run_name / f"epoch_{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            prefix_state_dict = unwrapped_model.prefix_embeddings.state_dict()
            torch.save(prefix_state_dict, ckpt_dir / "prefix_embeddings.pt")
            
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

        
        scheduler.step()

    if accelerator.is_main_process:
        print("Training finished.")
        # Final save
        final_dir = Path(args.save_root) / run_name / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        prefix_state_dict = unwrapped_model.prefix_embeddings.state_dict()
        torch.save(prefix_state_dict, final_dir / "prefix_embeddings.pt")
        tokenizer.save_pretrained(final_dir)
        print(f"Final prefix saved to {final_dir}")
        
    accelerator.end_training()


if __name__ == "__main__":
    main()
