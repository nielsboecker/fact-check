#!/usr/bin/env bash
# Note: Run this with python 2.7 and tensorflow 1.5 (use conda environment if possible)

# Train
python src/SentenceMatchTrainer.py --config_path configs/fever.bimpm.config

# Evaluate
python src/SentenceMatchDecoder.py \
    --in_path data/fever/dev.tsv \
    --word_vec_path data/GoogleNews-vectors-negative300-SLIM.txt \
    --out_path ../generated/entailment_baseline_evaluation_result.json \
    --model_prefix data/logs/SentenceMatch.fever
