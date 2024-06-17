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