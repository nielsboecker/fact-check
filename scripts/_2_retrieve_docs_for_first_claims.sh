#!/usr/bin/env bash

# This script can be used to retrieve documents with multiple retrieval approaches
# one after the other. The first argument specifies which dataset to use. If it is
# not adjusted, it will only work with the top 15 claims (due to --limit flag).

DATASET=$1

python3 src/_2_F_retrieve_docs_with_tfidf.py --dataset ${DATASET} --limit
echo ">>>>> TFIDF retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --dataset ${DATASET} --limit
echo ">>>>> Bare query likelihood retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --dataset ${DATASET} --limit --smoothing "laplace"
echo ">>>>> laplace retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --dataset ${DATASET} --limit --smoothing "laplace_lindstone"
echo ">>>>> laplace_lindstone retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --dataset ${DATASET} --limit --smoothing "jelinek_mercer"
echo ">>>>> jelinek_mercer retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --dataset ${DATASET} --limit --smoothing "dirichlet"
echo ">>>>> dirichlet retrieval finished"
