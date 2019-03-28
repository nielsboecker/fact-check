#!/bin/bash

python3 src/_2_F_retrieve_docs_with_tfidf.py --limit
echo ">>>>> TFIDF retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --limit
echo ">>>>> Bare query likelihood retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --limit --smoothing "laplace"
echo ">>>>> laplace retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --limit --smoothing "laplace_lindstone"
echo ">>>>> laplace_lindstone retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --limit --smoothing "jelinek_mercer"
echo ">>>>> jelinek_mercer retrieval finished"

python3 src/_3_F_retrieve_docs_with_unigram_query_likelihood.py --limit --smoothing "dirichlet"
echo ">>>>> dirichlet retrieval finished"

