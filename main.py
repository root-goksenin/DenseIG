from denseig.utils import summarize_attributions, visualize, _write_html, concat_title_and_body, calculate_attributions
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers.util import dot_score
import torch 

import random 
import click 

random.seed(42)



@click.command()
@click.argument("path_to_beir_data")
@click.argument("base_model", default = "GPL/msmarco-distilbert-margin-mse")
@click.argument("adapted_model", default = "GPL/trec-covid-msmarco-distilbert-gpl")
def main(path_to_beir_data, base_model, adapted_model):
    corpus, queries, qrels = GenericDataLoader(path_to_beir_data).load("test")
    
    qid = random.choice(list(queries.keys()))
    query = [queries[qid]]        
    did = list(qrels[qid])[0]
    doc = [concat_title_and_body(did, corpus)]
    calculate_attributions(model_before = base_model,
        model_after = adapted_model,
        query= query,
        doc = doc,
        scoring_function = dot_score,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
if __name__ == "__main__":
    main()