import sys
import os
from transformers import (
    AutoModel, 
    AutoConfig, 
    PushToHubCallback,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
    
)

model_ckpt_path = sys.argv[1]
serial = sys.argv[2]
model_type = sys.argv[3]
model_name = sys.argv[4]

def main():

    if model_type == 'decoder':
        model = AutoModelForCausalLM.from_pretrained(model_ckpt_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt_path)
    
    model.push_to_hub(model_name, use_auth_token=True, from_tf=True)

if __name__ == "__main__":
    exit(main())