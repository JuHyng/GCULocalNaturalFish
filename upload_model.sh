

export serial="20231002_053021"
export checkpoint="28800"

export model_name="$serial-$checkpoint"



huggingface-cli repo create $model_name --private
huggingface-cli login 

echo !!Upload HuggingFace $serial checkpoint $checkpoint

python upload_model.py \
     '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
    $checkpoint \
    "encoder-decoder" \
    $model_name

export serial="20231002_053021"
export checkpoint="34200"

export model_name="$serial-$checkpoint"



huggingface-cli repo create $model_name --private

echo !!Upload HuggingFace $serial checkpoint $checkpoint

python upload_model.py \
     '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
    $checkpoint \
    "encoder-decoder" \
    $model_name

export serial="20231002_053021"
export checkpoint="45000"

export model_name="$serial-$checkpoint"



huggingface-cli repo create $model_name --private

echo !!Upload HuggingFace $serial checkpoint $checkpoint

python upload_model.py \
     '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
    $checkpoint \
    "encoder-decoder" \
    $model_name






