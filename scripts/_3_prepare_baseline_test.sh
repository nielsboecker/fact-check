#!/usr/bin/env bash

# Clone my BiMPM fork incl. parse error fix
git clone git@github.com:nbckr/BiMPM.git
cd BiMPM

mkdir -p data
cd data

# Download slim Word2Vec embeddings
W2V_NAME=GoogleNews-vectors-negative300-SLIM
wget https://github.com/eyaler/word2vec-slim/raw/master/${W2V_NAME}.bin.gz
gunzip ${W2V_NAME}.bin.gz

# Convert bin to txt embeddings
git clone git@github.com:marekrei/convertvec.git
cd convertvec
make
./convertvec bin2txt ../${W2V_NAME}.bin ../${W2V_NAME}.txt
cd ..

# Copy preprocessed data
mkdir -p fever
cp ../../generated/entailment_baseline_preprocessed_train.tsv fever/train.tsv
cp ../../generated/entailment_baseline_preprocessed_dev.tsv fever/dev.tsv

# Copy config
cd ..
mkdir -p configs
cp ../submission/bimpm_baseline/fever.bimpm.config configs/
