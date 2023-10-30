export checkpoint='4450'
export serial='20231013_142822'
export model_name='망치상어'

export model_ckpt_path='togru'/$serial-$checkpoint
export model='nlpai-lab/kullm-polyglot-12.8b-v2'
export date='validation'

export prompt="### 명령어:\n두 문장 사이에 생략된 문장을 생성하시오\n\n### 입력:\nsent1 sent3\n\n### 응답:\nlabel"

echo !!INFERENCE $serial checkpoint $checkpoint


export file_name=$model_name'-'$checkpoint'-1'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1



export file_name=$model_name'-'$checkpoint'-2'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1



export file_name=$model_name'-'$checkpoint'-3'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1



export checkpoint='4270'
export model_ckpt_path='togru'/$serial-$checkpoint

export file_name=$model_name'-'$checkpoint'-1'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1


export file_name=$model_name'-'$checkpoint'-2'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1


export file_name=$model_name'-'$checkpoint'-3'
python inference.py \
    --model_ckpt_path $model_ckpt_path \
    --tokenizer $model \
    --output_jsonl ./submission/$date/$file_name.jsonl \
    --model-path $model \
    --classifier_hidden_size 4096 \
    --prompt "$prompt" \
    --decoder True \
    --max-seq-len 256 \
    --batch-size 16 \
    --num_return_sequences 1



export model='paust/pko-t5-large'
export date='validation'
export serial="20231009_185511"
# export prompt="다음 두 문장 사이에 순서 상 생략된 문장을 생성하시오: sent1 <extra_id_0> sent3"
export prompt="문장1과 문장2 사이에 생략된 문장을 생성하시오\n문장1:sent1\n<extra_id_0>\n문장2:sent3"


for checkpoint in '32400'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint


    export model_name='돌아온_은갈치-'$checkpoint'-nb8'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 16 \
        --num_beams 8 \
        --num_return_sequences 1


    
    export model_name='돌아온_은갈치-'$checkpoint'-k10p92'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 8 \
        --do_sample True \
        --top_k 10 \
        --top_p 0.92 \
        --num_return_sequences 1

    export model_name='돌아온_은갈치-'$checkpoint'-nb16'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 8 \
        --num_beams 16 \
        --num_return_sequences 1
done

for checkpoint in '45000'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint

    export model_name='돌아온_은갈치-'$checkpoint'-nb12'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 8 \
        --num_beams 12 \
        --num_return_sequences 1

done

export serial='20231002_053021'
export model='paust/pko-t5-large'
export prompt="문장1과 문장2 사이에 생략된 문장을 생성하시오\n문장1:sent1\n<extra_id_0>\n문장2:sent3"
export data='validation'

for checkpoint in '28800'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint

    export model_name='또다시_보리숭어-'$checkpoint'-nb8'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 16 \
        --num_beams 8 \
        --num_return_sequences 1
done

for checkpoint in '32400'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint

    export model_name='또다시_보리숭어-'$checkpoint'-nb8'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 16 \
        --num_beams 8 \
        --num_return_sequences 1
done

for checkpoint in '34200'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint

    export model_name='또다시_보리숭어-'$checkpoint'-nb12'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 16 \
        --num_beams 12 \
        --num_return_sequences 1
done

for checkpoint in '45000'
do
    echo !!INFERENCE $serial checkpoint $checkpoint
    export model_ckpt_path='togru'/$serial-$checkpoint

    export model_name='또다시_보리숭어-'$checkpoint'-nb12'
    python inference.py \
        --model_ckpt_path $model_ckpt_path \
        --tokenizer $model \
        --output_jsonl ./submission/$date/$model_name.jsonl \
        --model-path $model \
        --classifier_hidden_size 1024 \
        --prompt "$prompt" \
        --batch-size 16 \
        --num_beams 12 \
        --num_return_sequences 1
done