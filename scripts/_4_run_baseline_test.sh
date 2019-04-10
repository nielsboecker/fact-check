#!/usr/bin/env bash
# Note: Run this with python 2.7 and tensorflow 1.5 (use conda environment if possible)
# wget https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh
# bash Anaconda2-2019.03-Linux-x86_64.sh
# conda create --name fever python=2.7
# conda activate fever
# conda install tensorflow=1.5

cd BiMPM/

# Train
python src/SentenceMatchTrainer.py --config_path configs/fever.bimpm.config

# Evaluate
python src/SentenceMatchDecoder.py \
    --in_path data/fever/dev.tsv \
    --word_vec_path data/GoogleNews-vectors-negative300-SLIM.txt \
    --out_path ../generated/entailment_baseline_evaluation_result.json \
    --model_prefix data/logs/SentenceMatch.fever

cd ..

