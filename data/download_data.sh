#!/bin/bash

# Pre-processed Wikipedia Pages (June 2017 dump)
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip &&
    unzip wiki-pages.zip &&
    rm -r __MACOSX &&
    rm licence.html &&
    rm wiki-pages.zip

# Shared Task Development Dataset (Labelled)
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl

# Shared Task Blind Test Dataset (Unlabelled)
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl

# Training Dataset
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl

echo "Downloaded all input data"
