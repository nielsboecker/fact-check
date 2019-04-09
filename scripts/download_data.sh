#!/bin/bash

# Supposed to be run from root project directory
mkdir -p data
cd data

# Pre-processed Wikipedia Pages (June 2017 dump)
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip &&
    unzip wiki-pages.zip &&
    rm -r __MACOSX &&
    rm license.html &&
    rm wiki-pages.zip

# Shared Task Development Dataset (Labelled)
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl

# Shared Task Blind Test Dataset (Unlabelled)
wget https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl

# Training Dataset
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl

# Word2Vec Pre-trained word and phrase vectors
# VECTORS_FILE=GoogleNews-vectors-negative300.bin.gz
# gdrive_download 0B7XkCwpI5KDYNlNUTTlSS21pQmM ${VECTORS_FILE} &&
# gunzip ${VECTORS_FILE} &&
# rm -f ${VECTORS_FILE}

# GloVe pre-trained word vectors
GLOVE_FILE_NAME="glove.840B.300d"
wget http://nlp.stanford.edu/data/${GLOVE_FILE_NAME}.zip
unzip ${GLOVE_FILE_NAME}.zip
rm -rf ${GLOVE_FILE_NAME}.zip
# Transform GloVe to Word2Vec format, refer to https://radimrehurek.com/gensim/scripts/glove2word2vec.html
sed -i '1i2196017 300' ${GLOVE_FILE_NAME}.txt


echo "Downloaded all input data"
cd ..


# Source: https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
# function gdrive_download () {
#   CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
#   wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
#   rm -rf /tmp/cookies.txt
# }
