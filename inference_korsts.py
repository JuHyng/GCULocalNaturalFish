import json
import sys
import torch
from torch import nn
from transformers import ElectraConfig
from transformers import ElectraModel, AutoTokenizer, ElectraTokenizer, ElectraForSequenceClassification


max_seq_length = 128
tokenizer = AutoTokenizer.from_pretrained("daekeun-ml/koelectra-small-v3-korsts")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Huggingface pre-trained model: 'monologg/koelectra-small-v3-discriminator'
def model_fn(model_path=None):
    ####
    # If you have your own trained model
    # Huggingface pre-trained model: 'monologg/koelectra-small-v3-discriminator'
    ####    
    #config = ElectraConfig.from_json_file(f'{model_path}/config.json')
    #model = ElectraForSequenceClassification.from_pretrained(f'{model_path}/model.pth', config=config)
    model = ElectraForSequenceClassification.from_pretrained('daekeun-ml/koelectra-small-v3-korsts')
    model.to(device)
    return model


def input_fn(input_data, content_type="application/jsonlines"):
    # data_str = input_data.decode("utf-8")
    # jsonlines = data_str.split("\n")
    transformed_inputs = []
    
    text = input_data["text"]
    # logger.info("input text: {}".format(text))          
    encode_plus_token = tokenizer.encode_plus(
        text,
        max_length=max_seq_length,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    transformed_inputs.append(encode_plus_token)
        
    return transformed_inputs


def predict_fn(transformed_inputs, model):
    predicted_classes = []

    for data in transformed_inputs:
        data = data.to(device)
        output = model(**data)

        prediction_dict = {}
        prediction_dict['score'] = output[0].squeeze().cpu().detach().numpy().tolist()

    return prediction_dict


def output_fn(outputs, accept="application/jsonlines"):
    return outputs, accept
