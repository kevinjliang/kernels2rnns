#!/bin/bash

GPU=$1
MODEL=$2
NGRAM=$3
DATASETS=("agnews" "dbpedia" "yelp_full" "yahoo")

for i in ${!DATASETS[*]}
do
    echo "CUDA_VISIBLE_DEVICES=$GPU python main.py --output_dir './Outputs/$MODEL/${DATASETS[$i]}/' --dataset=${DATASETS[$i]} --model $MODEL --ngram_length $NGRAM --layer_norm"

    CUDA_VISIBLE_DEVICES=$GPU python main.py --output_dir ./Outputs/$MODEL/${DATASETS[$i]}/ --dataset=${DATASETS[$i]} --model $MODEL --ngram_length $NGRAM --layer_norm
done
