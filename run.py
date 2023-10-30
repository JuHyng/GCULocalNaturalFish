import os
import sys
import torch
import wandb
import numpy as np

from torch.utils.data import DataLoader
from functools import partial

#module import 부분
from modules.logger_module import get_logger, log_args, setup_seed
from modules.utils import create_output_dir, load_tokenizer, load_datasets, get_labels_from_dataset
from modules.arg_parser import get_args
from modules.dataset_preprocessor import load_and_process_datasets

from konlpy.tag import Mecab
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score_func
# from bleurt import score

#model import 
from models.KoBART import KoBART
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PrefixTuningConfig, TaskType

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EvalPrediction,
    AdamW,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    DataCollatorForLanguageModeling
    
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mecab = Mecab()

# checkpoint = "BLEURT-20"

# scorer = score.BleurtScorer(checkpoint)

class DecoderDataCollator:
    def __init__(self, tokenizer, response):
        self.tokenizer = tokenizer
        self.response = response

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        max_length = max([len(ids) for ids in input_ids])

        # Padding
        input_ids = [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        input_ids = torch.tensor(input_ids)

        # Create labels
        labels = input_ids.clone().detach()
        
        # Convert the response token to its corresponding IDs (might be more than one token)
        response_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.response))

        for label in labels:
            # Find the start position of the response token sequence
            response_start = -1
            for i in range(len(label) - len(response_token_ids) + 1):
                if all(label[i+j] == response_token_ids[j] for j in range(len(response_token_ids))):
                    response_start = i
                    break

            if response_start != -1:
                label[:response_start] = -100

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

def main(args):
    #wandb option args 이용할것

    
    # Setup logger
    logger = get_logger("train")
    log_args(args, logger)
    setup_seed(args.seed, logger)

    # Create output directory and log the action
    
    logger.info(create_output_dir(args.output_dir))

    # Log the GPU information
    logger.info(f"[+] GPU: {args.gpus}")

    # Load tokenizer and log the action
    tokenizer = load_tokenizer(args.tokenizer)
    logger.info(f'[+] Loaded Tokenizer')

    # # Load datasets and log the action    
    if args.decoder is False:
        encoded_train_ds, encoded_valid_ds = load_and_process_datasets(tokenizer, args.train_path, args.val_path, args.max_seq_len, args.prompt, args.decoder)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        encoded_train_ds, encoded_valid_ds = load_and_process_datasets(tokenizer, args.train_path, args.val_path, args.max_seq_len, args.prompt, args.decoder)
        
    logger.info(f'[+] Loaded Dataset')
    logger.info(encoded_train_ds)
    
    
    if args.quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        #PEFT: lora
        if args.peft == 'lora':
            
            if not args.decoder:
                task_type = TaskType.SEQ_2_SEQ_LM
            else:
                task_type = TaskType.CAUSAL_LM
            
            config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none", 
            task_type=task_type
            )
            if args.decoder is False:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, quantization_config=bnb_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config)
          
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)

            model = get_peft_model(model, config)
        
        #prefix-tuning (error when quant True)
        elif args.peft == 'prefix':
            
            if not args.decoder:
                task_type = TaskType.SEQ_2_SEQ_LM
            else:
                task_type = TaskType.CAUSAL_LM
                
            config = PrefixTuningConfig(
                task_type=task_type,
                inference_mode=False, 
                num_virtual_tokens=args.virtual_len)
            
            if args.decoder is False:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, quantization_config=bnb_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config)

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            
            model.print_trainable_parameters()
            logger.info(model.print_trainable_parameters())
        
        #quant True, peft None
        else:
            
            if args.decoder is False:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, quantization_config=bnb_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb_config)
            
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            
    else:
        if args.decoder is False:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
                
    train_and_validate(args,tokenizer, model, encoded_train_ds, encoded_valid_ds)

def initialize_wandb(args, fold):
    wandb_runname = "{}_{}_{}".format(args.peft, args.model_path, args.output_dir.split('/')[2])
    wandb.init(name=wandb_runname,    
               project=args.wandb_project)
    wandb.config.update(args)


def compute_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 단순 후처리
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    metrics = evaluate_generation_metrics(decoded_preds, decoded_labels)
    return metrics

# def calc_bleurt(true_data, pred_data):
#     if type(true_data[0]) is list:
#         true_data = list(map(lambda x: x[0], true_data))

#     scores = scorer.score(references=true_data, candidates=pred_data)

#     return sum(scores) / len(scores)

def calc_ROUGE_1(true, pred):
    # rouge_evaluator = Rouge(
    #     metrics=["rouge-n", "rouge-l"],
    #     max_n=2,
    #     limit_length=True,
    #     length_limit=1000,
    #     length_limit_type="words",
    #     apply_avg=True,
    #     apply_best=False,
    #     alpha=0.5,  # Default F1_score
    #     weight_factor=1.0,
    # )
    rouge_evaluator = Rouge()
    scores = rouge_evaluator.get_scores(pred, true, avg=True)
    return scores['rouge-1']['f']

def calc_BLEU(true, pred, apply_avg=True, apply_best=False, use_mecab=True):
    stacked_bleu = []

    if type(true[0]) is str:
        true = list(map(lambda x: [x], true))

    for i in range(len(true)):
        best_bleu = 0
        sum_bleu = 0
        for j in range(len(true[i])):

            if use_mecab:
                ref = mecab.morphs(true[i][j])
                candi = mecab.morphs(pred[i])
            else:
                ref = true[i][j].split()
                candi = pred[i].split()


            score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))

            sum_bleu += score
            if score > best_bleu:
                best_bleu = score

        avg_bleu = sum_bleu / len(true[i])
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

    return sum(stacked_bleu) / len(stacked_bleu)

def calc_bertscore(true, pred):
    P, R, F1 = bert_score_func(cands=pred, refs=true, lang="ko", model_type='bert-base-multilingual-cased', rescale_with_baseline=True)
    return F1.mean().item()

def evaluate_generation_metrics(predictions, references):
    metrics = {}
    
    metrics["ROUGE-1"] = calc_ROUGE_1(references, predictions)
    metrics["BLEU"] = calc_BLEU(references, predictions)
    metrics["BERTScore"] = calc_bertscore(references, predictions)
    # metrics["BLEURT"] = calc_bleurt(references, predictions)
    
    metrics["AVERAGE"] = (metrics["ROUGE-1"] + metrics["BLEU"] + metrics["BERTScore"]) / 3
    # metrics["AVERAGE"] = (metrics["ROUGE-1"] + metrics["BLEURT"] + metrics["BERTScore"]) / 3
    
    return metrics

def train_and_validate(args, tokenizer, model, train_loader, val_loader, fold=None):
    initialize_wandb(args, None)
    
    output_dir = args.output_dir if fold is None else os.path.join(args.output_dir, f"fold_{fold+1}")
    
    if args.decoder is False:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            evaluation_strategy="steps",   # <-- "steps"로 변경
            logging_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_strategy="steps",   # <-- "steps"로 변경
            save_steps=args.logging_steps,   # <-- 몇 스텝마다 저장할지 지정. logging_steps와 같게 설정했지만 다르게 설정할 수도 있습니다.
            load_best_model_at_end=True,
            metric_for_best_model="AVERAGE",  # f1 점수를 기준으로 가장 좋은 모델을 선택
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            predict_with_generate=True,
            generation_max_length=20,
            generation_num_beams=args.num_beams            
        )

        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        bound_compute_metrics = partial(compute_metrics, tokenizer)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=args.callbacks)],
            train_dataset=train_loader,
            eval_dataset=val_loader,
            tokenizer=tokenizer, 
            data_collator=data_collator,
            compute_metrics=bound_compute_metrics,
            optimizers=(optimizer, scheduler)
        )
    
    #decoder
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            # evaluation_strategy="steps",   # <-- "steps"로 변경
            logging_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_strategy="steps",   # <-- "steps"로 변경
            save_steps=args.logging_steps,   # <-- 몇 스텝마다 저장할지 지정. logging_steps와 같게 설정했지만 다르게 설정할 수도 있습니다.
            # load_best_model_at_end=True,
            # metric_for_best_model="AVERAGE", 
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            #predict_with_generate=True
        )
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs)
        # bound_compute_metrics = partial(compute_metrics, tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_loader,
            # eval_dataset=val_loader,
            tokenizer=tokenizer, 
            data_collator=DecoderDataCollator(tokenizer, args.response),
            # compute_metrics=bound_compute_metrics,  # 평가 지표 계산 함수 전달
            optimizers=(optimizer, scheduler)
        )
        

        model.config.use_cache = False

    trainer.train()
    
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    args = get_args()
    
    exit(main(args))
