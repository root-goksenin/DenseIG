from denseig.wrapers import SentenceTransformerWrapper 
from denseig.models import DocModel, QueryModel
from .utils import load_sbert
from captum.attr import LayerIntegratedGradients

def _get_input_features(query, doc, model, device):
    parts_query = SentenceTransformerWrapper(load_sbert(model), device)
    parts_doc = SentenceTransformerWrapper(load_sbert(model), device)
    query_input, query_ref, query_tokens = parts_query.return_text_and_base_features(query) 
    doc_input, doc_ref, doc_tokens =  parts_doc.return_text_and_base_features(doc) 
    
    # Generate query and ref
    query_input_ids, query_attention_mask = query_input['input_ids'], query_input['attention_mask']
    query_ref_ids, _ = query_ref['input_ids'], query_ref['attention_mask']
    
    # Generate doc and ref
    doc_input_ids, doc_attention_mask = doc_input['input_ids'], doc_input['attention_mask']
    doc_ref_ids, _ = doc_ref['input_ids'], doc_ref['attention_mask']    
    return parts_query, parts_doc, query_tokens, query_input_ids, query_attention_mask, query_ref_ids, doc_tokens, doc_input_ids, doc_attention_mask, doc_ref_ids


def generate_attributions_query(query, doc, model, scoring_function, device):
    parts_query, parts_doc, query_tokens, query_input_ids, query_attention_mask, query_ref_ids, _, doc_input_ids, doc_attention_mask, _= _get_input_features(query, doc, model, device)
    model_q = QueryModel(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler, doc_input_ids, doc_attention_mask,
                                scoring_function = scoring_function)
    # Get the query model embeddings
    ig_q = LayerIntegratedGradients(model_q, model_q.query_model.embeddings)
    return  query_tokens, ig_q.attribute(inputs = query_input_ids, 
                        baselines = query_ref_ids,
                        internal_batch_size = 1,
                        additional_forward_args = (query_attention_mask),
                        n_steps = 700,
                        return_convergence_delta=True
                        )
    
    
def generate_attributions_doc(query, doc, model, scoring_function, device):
    parts_query, parts_doc, _, query_input_ids, query_attention_mask, _, doc_tokens, doc_input_ids, doc_attention_mask, doc_ref_ids = _get_input_features(query, doc, model, device)
    model_d = DocModel(parts_query.bert_model, parts_doc.bert_model, parts_query.pooler, query_input_ids, query_attention_mask,
                            scoring_function = scoring_function)
    # Get the doc model embeddings.
    ig_d = LayerIntegratedGradients(model_d, model_d.doc_model.embeddings)
    return  doc_tokens, ig_d.attribute(inputs = doc_input_ids, 
                        baselines = doc_ref_ids,
                        internal_batch_size = 1,
                        additional_forward_args = (doc_attention_mask),
                        n_steps = 700,
                        return_convergence_delta=True
                        )

      
