import torch
from transformers import AutoTokenizer
from datasets import Dataset
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm



def preprocess(tokenizer, max_seq_len, example, prompt, decoder, is_training):
    processed = {}
    sentence_1 = example["input"]["sentence1"]
    sentence_3 = example["input"]["sentence3"]
    label = example["output"]
    
        
    prompt = prompt.replace("sent1", sentence_1)
    prompt = prompt.replace('sent3', sentence_3)
    
    # if tokenizer.sep_token is None:
    #     prompt = prompt.replace('sep', ' ')
    # else:
    #     prompt = prompt.replace('sep', tokenizer.sep_token)
        
    if not decoder or is_training is False:
        if ('label' in prompt):
            prompt = prompt.replace("label", '')
    else:
        prompt = prompt.replace('label', label)
    
    tokenizer_input = tokenizer(
        prompt,
        padding="max_length",
        add_special_tokens=True, 
        max_length=max_seq_len,
        truncation=True
    )
    
    processed["input_ids"] = tokenizer_input["input_ids"]
    processed["attention_mask"] = tokenizer_input["attention_mask"]
    
    

    if decoder is False:
        
        tokenizer_output = tokenizer(
        example["output"],
        add_special_tokens=True, 
        padding="max_length",
        max_length=max_seq_len,
        truncation=True
        )
        processed['labels'] = tokenizer_output["input_ids"]
        processed["decoder_input_ids"] = tokenizer_output["input_ids"]
        
        if is_training:
            processed["decoder_attention_mask"] = tokenizer_output["attention_mask"]

    return processed

def load_and_process_datasets(tokenizer, train_path, val_path, max_seq_len, prompt, decoder):
    train_dataset = Dataset.from_json(train_path)
    encoded_train_dataset = train_dataset.map(lambda x: preprocess(tokenizer, max_seq_len, x, prompt, decoder, is_training=True,), )

    if val_path:
        val_dataset = Dataset.from_json(val_path)
        encoded_val_dataset = val_dataset.map(lambda x: preprocess(tokenizer, max_seq_len, x, prompt, decoder, is_training=False), )
    else:
        encoded_val_dataset = None

    return encoded_train_dataset, encoded_val_dataset

# def preprocess_decoder(tokenizer, max_seq_len, example, choose_prompt ,is_training):
#     processed = {}
#     sentence_1 = example["input"]["sentence1"]
#     sentence_2 = example["input"]["sentence3"]
#     label = example["output"]
    
#     # prompt_list = [
#     #     f'{sentence_1} {tokenizer.sep_token} {sentence_2}',
#     #     f'{sentence_1} {tokenizer.sep_token} {sentence_2} 사이에 들어갈 문장을 생성하시오',
#     #     f'앞의 문장과 뒤의 문장 사이에 논리적으로 연결하는 문장을 생성하세요: {sentence_1} {tokenizer.sep_token} {sentence_2}.',
#     #     f'앞에 문장: {sentence_1}. 뒤에 문장: {sentence_2}. 중간에 연결되는 문장은?"',
#     #     f'앞의 문장: {sentence_1}. 뒤의 문장: {sentence_2}. 중간의 문장은?',
#     #     f'다음은 두 개의 문장입니다: {sentence_1}. {sentence_2}. 이 두 문장 사이에 어떤 문장이 올까요?',
#     #     f"두 문장, {sentence_1}와 {sentence_2}, 사이의 빠진 문장은?",
#     #     ]
    
#     answer_prompt_list = [
#         f'\n{label}',
        
#     ]
    
#     if is_training is True:
#         prompt = prompt_list[choose_prompt]+ answer_prompt_list
#     else:
#         prompt = prompt_list[choose_prompt]

#     tokenizer_input = tokenizer(
#         prompt,
#         padding="max_length",
#         max_length=max_seq_len,
#         truncation=True
#     )
#     processed["input_ids"] = tokenizer_input["input_ids"]
#     processed["attention_mask"] = tokenizer_input["attention_mask"]
    

#     return processed

# def load_and_process_datasets_decoder(tokenizer, train_path, val_path, max_seq_len):
#     train_dataset = Dataset.from_json(train_path)
#     encoded_train_dataset = train_dataset.map(lambda x: preprocess_decoder(tokenizer, max_seq_len, x, is_training=True))

#     if val_path:
#         val_dataset = Dataset.from_json(val_path)
#         encoded_val_dataset = val_dataset.map(lambda x: preprocess_decoder(tokenizer, max_seq_len, x, is_training=False), load_from_cache_file=False)
#     else:
#         encoded_val_dataset = None

#     return encoded_train_dataset, encoded_val_dataset

