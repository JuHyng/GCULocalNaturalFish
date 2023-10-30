export date='validation'

for checkpoint in '4270' '4450'
do
    for num in '1' '2' '3'
    do
        python clean_decoder_result.py \
            ./submission/$date/망치상어-$checkpoint-$num.jsonl \
            '### 응답:\\n'
    done
done