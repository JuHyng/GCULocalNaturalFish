wandb login 29eecad670264cabb466a26fd65dd9ddb23931ee
export CUDA_VISIBLE_DEVICES=0

# for model in  paust/pko-flan-t5-large
# do
#     for prompt in "sent1 sent3 두 문장 사이에 생략된 문장을 생성하시오"
#     do
#     python run.py --model-path $model --tokenizer $model \
#         --wandb_project MODU_SC \
#         --batch-size 8 \
#         --valid_batch_size 4 \
#         --classifier_hidden_size 1024 \
#         --learning-rate 2e-5 \
#         --epochs 10 \
#         --prompt "$prompt" \
#         --peft ft \

#     done
# done

# for model in  paust/pko-t5-large
# do
#     for prompt in "\"문장2\"는 \"문장1\"의 결과이며 \"문장3\"의 원인이다. \"문장1\"과 \"문장3\" 사이에 순서 상 생략된 \"문장2\"를 생성하시오: \"문장1\":sent1 \"문장2\":<extra_id_0> \"문장3\":sent3" 
#     do
#     python run.py --model-path $model --tokenizer $model \
#         --wandb_project MODU_SC \
#         --batch-size 8 \
#         --valid_batch_size 1 \
#         --classifier_hidden_size 1024 \
#         --learning-rate 5e-5 \
#         --epochs 10 \
#         --prompt "$prompt" \
#         --peft ft \
#         --callbacks 100 \
#         --logging_steps 1800 \
#         --seed 777 \
#         --train_path ./data/train+dev.jsonl \
#         --val_path ./submission/ensemble/세꼬시7.jsonl

#     done
# done


# for model in nlpai-lab/kullm-polyglot-5.8b-v2
# do
#     for prompt in "### 질문: sent1 sent3 ### 응답: sent1 label sent3"
#     do
#     python run.py --model-path $model --tokenizer $model \
#         --wandb_project MODU_SC \
#         --batch-size 128 \
#         --classifier_hidden_size 4096 \
#         --peft lora \
#         --quant True \
#         --learning-rate 3e-4 \
#         --epochs 2 \
#         --logging_steps 10 \
#         --decoder True \
#         --train_path ./data/train+dev.jsonl \
#         --prompt "$prompt"
#     done
# done


# "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n"
for model in EleutherAI/polyglot-ko-12.8b
do
    for prompt in "### 명령어:\n\"문장1\"과 \"문장2\" 사이에 순서 상 생략된 문장을 생성하시오\n\n### 입력:\n\"문장1\": sent1 \"문장2\": sent3\n\n### 응답:\nlabel"
    do
    python run.py --model-path $model --tokenizer $model \
        --wandb_project MODU_SC \
        --batch-size 64 \
        --classifier_hidden_size 4096 \
        --max-seq-len 128 \
        --peft lora \
        --quant True \
        --learning-rate 5e-4 \
        --epochs 7 \
        --logging_steps 10 \
        --decoder True \
        --train_path ./data/train+dev.jsonl \
        --prompt "$prompt"
    done
done