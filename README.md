# Course Project: Automated Fact Checking
Coursework for COMP0084 Information Retrieval and Data Mining
University College London, 2019


## Project Description
This project aims at exploiting information retrieval and data mining methods
to assess the veracity of claims made online. At its core, the project consists
of three steps:
- relevant document retrieval
- evidence sentence selection
- claim veracity prediction

The dataset used is the publicly available Fact Extraction and Verification 
(FEVER) dataset. The data is available in the following ways:
- `train.jsonl`: labeled training dataset
- `shared_task_dev.jsonl`: labeled development dataset
- `shared_task_dev_public.jsonl`: unlabeled development dataset
- `shared_task_test.jsonl`: unlabeled testing dataset


## How to reproduce
The scripts for all sub-tasks are in the `src/` directory. They are clearly named according to the scheme `_N_i_description.py`, where N is the sub-task, i defines the order within it, and the description summarises what the script does. To find out about configuration, invoke the script with `--help` flag.

As the scripts are often using the output of earlier scripts, i.e. first generating the inverted index and then using it for retrieval speed-up, they should all be run in sequence to reproduce the results. In addition, `scripts/_1_download_data.sh` should be run initially to download the corpus, word embeddings etc. Also, `scripts/_2_retrieve_docs_for_first_claims.sh` can be used for convenience to run all implemented retrieval variations for a small sample from the training set at once.

The solution was developed using Python3 and the libraries listed in `requirements.txt`. To reproduce, you could use conda:
```
# wget https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh
bash Anaconda2-2019.03-Linux-x86_64.sh
conda create --name fever python=3.7
conda activate fever
conda install --yes --file requirements.txt
```

The only exception is the BiMPM baseline, which is using Python 2. For reproducing that, further instruction can be found in the two relevant scripts in the `scrips/` directory.


## Directory Structure
```
.
├── data
├── generated
    # this directory will contain results of auxiliary steps, which scripts in /src will generate
│   ├── figures
        # generated figures for the report
│   ├── inverted_index
        # inverted index for retrieval tasks
│   └── wiki_page_batch_mappings
        # mapping to increase wiki_page retrieval
├── main_report
    # this directory contains the report that accompanies these experiments
├── retrieved
    # this directory will contain results of retrieval scripts in /src
    # the subdirectorys' name indicate dev/test/train subset, retrieval method, 
    # and the suffix 10000 indicates whether a large sample was used
│   ├── probabilistic_dev
│   │   └── laplace_lindstone_0.01_10000
│   ├── probabilistic_test
│   │   ├── dirichlet
│   │   └── laplace_lindstone
│   ├── probabilistic_train
│   │   ├── jelinek_mercer
│   │   ├── laplace
│   │   ├── laplace_lindstone_0.01
│   │   ├── laplace_lindstone_0.01_10000
│   │   ├── laplace_lindstone_0.1
│   │   └── no_smoothing
│   └── tfidf
├── scripts
    # this directory contains scripts to initialise a server for the processing, download the
    # necessary data, invoke the multiple retrieval implementations from /src, and conduct the
    # baseline test with BiMPM (see report)
├── src
    # this directory contains the python scripts for all subtasks, where the file name indicates the
    # respective task as well as the order of how to run the scripts to reproduce the results
    # most scripts support several flags to configue, information can be obtained by '--help'
    # subdirectories contain shared code submodules
│   ├── dataaccess
│   ├── documentretrieval
│   ├── model
│   ├── relevance
│   └── util
└── submission
    # this directory contains files that are mandatory to submit and other relevant artifacts
    ├── bimpm_baseline
        # configuration and result of running the dev subset through BiMPM
    ├── retrieved_dev
        # retrieved documents from dev subset
    └── retrieved_train
        # retrieved documents from train subset
        # subdirectories are using weighted scoring
        ├── title_lambda_0.25
        ├── title_lambda_0.5
        ├── title_lambda_0.75
        └── title_lambda_1.0
```