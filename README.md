# DenseIG
This repository contains the code for paper : [URL], and fully reproducible environment.

# Reproduce results 

First install the repo using:
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/DenseIG.git
    git checkout snapshot-reproduction
    ```


2. Install the project using pip install:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

To generate input attributions on BEIR datasets using the DenseIG models, and compare two models run the following command
```bash
   python3 main.py "path_to_beir_trec_covid_data" GPL/msmarco-distilbert-margin-mse GPL/trec-covid-msmarco-distilbert-gpl
```

```bash
   python3 main.py "path_to_beir_fiqa_data" GPL/msmarco-distilbert-margin-mse GPL/fiqa-msmarco-distilbert-gpl
```
