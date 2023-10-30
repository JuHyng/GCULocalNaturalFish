import os
import sys
import torch


import numpy as np
from datasets import Dataset
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from datasets import concatenate_datasets,Dataset
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import (AutoModel, AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    # Trainer,
    EvalPrediction,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from modules.logger_module import get_logger, log_args, setup_seed
from modules.utils import create_output_dir, load_tokenizer, load_datasets, get_labels_from_dataset
from modules.arg_parser import get_args
from modules.dataset_preprocessor import preprocess

from modules.utils import jsonldump, jsonlload
from tqdm import tqdm
from peft import PeftModel, PeftConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dataset):
        self.encoded_dataset = encoded_dataset

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        item = self.encoded_dataset[idx]
        input_ids = torch.tensor(item["input_ids"])
        attention_mask = torch.tensor(item["attention_mask"])
        return input_ids, attention_mask


def load_and_process_test_datasets(tokenizer, test_path, max_seq_len, prompt, decoder):
    test_dataset = Dataset.from_json(test_path)
    encoded_test_dataset = test_dataset.map(lambda x: preprocess(tokenizer, max_seq_len, x, prompt, decoder, is_training=False), )

    return encoded_test_dataset

def main(args):
    logger = get_logger("inference")
    log_args(args, logger)
    setup_seed(args.seed, logger)    
    np.random.seed(args.seed)
    
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    
    
    if args.decoder is False:
    
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_ckpt_path)
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        peft_model_id = args.model_ckpt_path
        model_id = args.model_path
        config = PeftConfig.from_pretrained(peft_model_id)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )       
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

        
    # 저장된 체크포인트로부터 가중치 로드
    # model.load_state_dict(torch.load(args.model_ckpt_path, map_location=device), strict=False)
    
    # 모델을 디바이스로 이동
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    
    tets_ds =  load_and_process_test_datasets(tokenizer, args.test_path, args.max_seq_len, args.prompt, args.decoder)
    tets_ds = CustomDataset(tets_ds)
    data_loader = DataLoader(tets_ds, batch_size=args.batch_size)

    total_summary_tokens = []
    for batch in tqdm(data_loader):
        input_ids, attention_mask = batch
        inputs = input_ids.to(device)
        
        attention_masks = attention_mask.to(device)
        if args.do_sample:
            summary_tokens = model.generate(
                input_ids=inputs,
                attention_mask=attention_masks,
                decoder_start_token_id=tokenizer.bos_token_id,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences
            )
        else:
            if args.decoder is False:
            
                summary_tokens = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_masks,
                    # temperature=0.001,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    max_length=64,
                    num_return_sequences=args.num_return_sequences
                )
                
            else:
                summary_tokens = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_masks,
                    temperature=0.001,
                    decoder_start_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # num_beams=args.num_beams,
                    max_new_tokens=64,
                    num_return_sequences=args.num_return_sequences
                )
        
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]
    
    j_list = jsonlload(args.test_path)
    for idx, oup in enumerate(decoded):
        # print(idx)
        
        # if args.decoder:
        #     oup = oup.replace(args.response,'')
        #     oup = oup.replace('문장1', '')
        #     oup = oup.replace('문장2', '')
        #     oup = oup.replace('\\n', '')
        
        j_list[idx]["output"] = oup

    jsonldump(j_list,args.output_jsonl)

if __name__ == "__main__":
    args = get_args()
    exit(main(args))
