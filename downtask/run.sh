#!/bin/bash

bszs=(64)   # (16 32 64)
lrs=(3e-5)  # (3e-5 5e-5 7e-5)
model_list=(pinsu/base_12)

for model in ${model_list[@]}; do
    for lr in ${lrs[@]}; do
        for bs in ${bszs[@]}; do
            python main.py  --df_file data/df_train.csv \
                    --epochs 10 \
                    --tokenizer ${model} \
                    --model_load_path ${model} \
                    --batch_size ${bs} \
                    --learning_rate ${lr}
        done
    done
done

# model_list=(microsoft/mdeberta-v3-base microsoft/deberta-v3-large)
# for model in ${model_list[@]}; do
#     python ner.py --tokenizer ${model} \
#                   --model_load_path ${model}
# done