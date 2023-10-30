import argparse
import torch
from datetime import datetime, timezone, timedelta

kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

def get_args():
    parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output_dir", type=str, default=f'./results/{train_serial}/', help="output directory path to save artifacts")
    g.add_argument("--valid_batch_size", type=int, default=1, help="validation batch size")
    g.add_argument("--epochs", type=int, default=10, help="the number of training epochs")
    g.add_argument("--learning-rate", type=float, default=2e-4, help="max learning rate")
    g.add_argument("--classifier_hidden_size", type=float, default=768, help="model hiddensize")
    g.add_argument("--classifier_dropout_prob", type=float, default=0.1, help="dropout rate")
    g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--train_path", type=str, default="./data/train.jsonl", help="train_set path")
    g.add_argument("--val_path", type=str, default="./data/dev.jsonl", help="validation_set path")
    g.add_argument("--test_path", type=str, default="./data/test.jsonl", help="test_set path")
    g.add_argument("--logging_steps", type=int, default="1800",help="logging_steps")
    g.add_argument("--model_ckpt_path", type=str, default="./results/20230821_205025/fold_5_best_model.pt", help="test_set path")
    g.add_argument("--output_jsonl", type=str, default=f'.test.jsonl', help="output directory path to save artifacts")
    
    g.add_argument("--response", type=str, default='### 응답:')
    g.add_argument("--decoder", type=bool, default=False, help="decoder")
    g.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v2", help="huggingface tokenizer path")
    g.add_argument("--model-path", type=str, default="gogamza/kobart-base-v2", help="model file path")
    
    g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradient accumulation steps")
    g.add_argument("--max-seq-len", type=int, default=128, help="max sequence length")
    g.add_argument("--batch-size", type=int, default=32, help="training batch size")    
    
    g.add_argument("--prompt", type=str, default="sent1 sep sent3 sep label", help="prompt template")
    g.add_argument("--callbacks", type=int, default=0, help="early callbacks")
    
    g = parser.add_argument_group("PEFT Options")
    g.add_argument("--virtual_len", type=int, default=0)
    g.add_argument("--peft", type=str, default='lora', help="Qlora")
    g.add_argument("--quant", type=bool, default=False)
    
    g = parser.add_argument_group("Wandb Options")
    g.add_argument("--wandb_run_name", type=str, default=f'{train_serial}',help="wanDB run name")
    g.add_argument("--wandb_entity", type=str, default='modu_ai' ,help="wanDB entity name")
    g.add_argument("--wandb_project", type=str, default='MODU_SC' ,help="wanDB project name")
    
    g = parser.add_argument_group("Inference Options")
    g.add_argument("--num_beams", type=int, default=8 ,help="number for beam_search")
    g.add_argument("--top_k", type=int, default=0 ,help="top_k sampling")
    g.add_argument("--do_sample", type=bool, default=False)
    g.add_argument("--top_p", type=float, default=1.0 ,help="top_p sampling")
    g.add_argument("--num_return_sequences", type=int, default=1 ,help="number of results to be returned")
    
    g.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    return parser.parse_args()
