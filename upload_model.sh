# export serial="20230921_233506"
# export checkpoint="25200"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      "./results/$serial/checkpoint-$checkpoint" \
#     $checkpoint \
#     "encoder-decoder" \
#     $model_name


# export serial="20231002_053021"
# export checkpoint="32400"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      "./results/$serial/checkpoint-$checkpoint" \
#     $checkpoint \
#     "encoder-decoder" \
#     $model_name

# export serial="20231013_142822"
# export checkpoint="4270"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      "./results/$serial/checkpoint-$checkpoint" \
#     $checkpoint \
#     "decoder" \
#     $model_name

# export serial="20231013_142822"
# export checkpoint="4450"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      "./results/$serial/checkpoint-$checkpoint" \
#     $checkpoint \
#     "decoder" \
#     $model_name

# export serial="20231009_185511"
# export checkpoint="32400"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
#     $checkpoint \
#     "encoder-decoder" \
#     $model_name

# export serial="20231009_185511"
# export checkpoint="45000"

# export model_name="$serial-$checkpoint"


# #token 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
# huggingface-cli repo create $model_name --private
# huggingface-cli login 

# echo !!Upload HuggingFace $serial checkpoint $checkpoint

# python upload_model.py \
#      '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
#     $checkpoint \
#     "encoder-decoder" \
#     $model_name

export serial="20231002_053021"
export checkpoint="28800"

export model_name="$serial-$checkpoint"


#token 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
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


#token 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
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


#token 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
huggingface-cli repo create $model_name --private

echo !!Upload HuggingFace $serial checkpoint $checkpoint

python upload_model.py \
     '/home/nlplab/hdd1/2023korean/korean_ai_2023/SC/results/'$serial'/checkpoint-'$checkpoint \
    $checkpoint \
    "encoder-decoder" \
    $model_name






