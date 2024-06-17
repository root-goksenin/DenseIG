# DenseIG

This repository contains the code for the DenseIG project.

## Introduction

DenseIG is a research project aimed at exploring explainability of the dense retriever techniques. The goal is to generate input attributions of dense retrievers using integrated gradients.
This project is aimed to explore the domain adapted model vs base model. However, you can taioler it to your needs.

## Installation via pip

To use the code in this repository, follow these steps:

1. Install via pypi

    ```bash
     pip install denseig
    ```

## Installation from source using poetry

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/DenseIG.git
    ```


2. Install the project using poetry install:

    ```bash
    poetry install
    ```

## Usage

To generate input attributions on BEIR datasets using the DenseIG models, and compare two models run the following command:
For now only two models are supported. This mainly serves the purpose of reproduction.

```bash
   python3 main.py "path_to_beir_data" "Non-adapted-dense-retriever" "adapted-dense-retriever"
```


To generate instance based attributions of any document-query pair, you should use the API. Only Dense Retriever models from SentenceTransformer framework is supported

```python
import torch 
from denseig.utils import generate_attributions_doc, generate_attributions_query, summarize_attributions
from sentence_transformers.util import dot_score
    
    
if __name__ == "__main__":
    query = ["What is python"]        
    doc = ["Python is an awsome tool"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only Dense Retriever models!
    model = "msmarco-distilbert-dot-v5"
    query_tokens, (q_attr, delta) = generate_attributions_query(query, doc, model, scoring_function = dot_score, device = device)
    q_attr  = summarize_attributions(q_attr)
    doc_tokens, (d_attr, delta) = generate_attributions_doc(query, doc, model, scoring_function = dot_score, device = device)
    doc_attr =  summarize_attributions(d_attr)
    print(q_attr)
```