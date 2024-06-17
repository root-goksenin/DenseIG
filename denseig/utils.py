import torch
import os 
from denseig.ig import generate_attributions_query, generate_attributions_doc
from .colored_terminal import bcolors
from captum.attr import visualization as viz
from typing import Dict, List



def _print_delta(delta):
    print(bcolors.WARNING + f"Delta is {delta.item()}" + bcolors.ENDC )

def _save_attr(q_attr, query_tokens, doc_attr, doc_tokens, file):
    query_vis = visualize(q_attr,query_tokens, title =  f"Query")
    doc_vis = visualize(doc_attr, doc_tokens,  title =  f"Document")
    html = viz.visualize_text([query_vis, doc_vis])
    _write_html(file, html.data)
    
    
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def _write_html(path, html):
    if os.path.split(path)[0]:
        os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w') as file:
        file.write(html)

def visualize(attr,tokens, title):    
    return viz.VisualizationDataRecord(
                        attr,
                        0,
                        0,
                        0,
                        title,
                        attr.sum(),       
                        tokens,
                        0)   
    
def calculate_attributions(model_before: str, model_after: str, query: str, doc: List[str], scoring_function, device):
    query_tokens, (q_attr, delta) = generate_attributions_query(query, doc, model_before,scoring_function = scoring_function, device = device)
    _print_delta(delta)
    q_attr  = summarize_attributions(q_attr)
    doc_tokens, (d_attr, delta) = generate_attributions_doc(query, doc, model_before, scoring_function = scoring_function, device = device)
    doc_attr =  summarize_attributions(d_attr)
    _print_delta(delta)
    _save_attr(q_attr, query_tokens, doc_attr, doc_tokens, file ="example_attributes/before_domain_adaptation_attr.html")
    
    
    query_tokens, (q_attr, delta) = generate_attributions_query(query, doc, model_after, scoring_function = scoring_function, device = device)
    q_attr  = summarize_attributions(q_attr)
    _print_delta(delta)
    doc_tokens, (d_attr,delta) = generate_attributions_doc(query, doc, model_after, scoring_function = scoring_function, device = device)
    doc_attr =  summarize_attributions(d_attr)
    _print_delta(delta)
    _save_attr(q_attr, query_tokens, doc_attr, doc_tokens, file ="example_attributes/after_domain_adaptation_attr.html")
   



def concat_title_and_body(did: str, corpus: Dict[str, Dict[str, str]], sep: str = " "):
  assert type(did) == str
  document = []
  title = corpus[did]["title"].strip()
  body = corpus[did]["text"].strip()
  if len(title):
      document.append(title)
  if len(body):
      document.append(body)
  return sep.join(document)