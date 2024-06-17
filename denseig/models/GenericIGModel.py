from torch import nn 

class IGModel(nn.Module):
    def __init__(self, query_model, doc_model, pooler, score):
        super(IGModel, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model
        self.score = score

    def forward_with_features(self, features, model):
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        trans_features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            trans_features.update({"all_layer_embeddings": hidden_states})
        
        return self.pooler(trans_features)['sentence_embedding']
    
    def forward(self, query_input_ids, doc_input_ids ,query_attention_mask, doc_attention_mask):
         # sourcery skip: inline-immediately-returned-variable
        
        features_query = {'input_ids': query_input_ids, 'attention_mask': query_attention_mask}
        features_doc = {'input_ids': doc_input_ids, 'attention_mask' : doc_attention_mask}
        q_emb = self.forward_with_features(features_query, self.query_model)
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = self.score(q_emb, doc_emb)
        return score