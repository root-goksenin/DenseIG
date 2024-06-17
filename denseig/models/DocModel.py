from torch import nn 

import torch.nn as nn

class DocModel(nn.Module):
    """
    A class representing a document model.

    Args:
        query_model (nn.Module): The query model.
        doc_model (nn.Module): The document model.
        pooler (nn.Module): The pooling layer.
        query_input_ids (torch.Tensor): The input IDs for the query.
        query_attention_mask (torch.Tensor): The attention mask for the query.
        scoring_function (callable): The scoring function to compute the similarity score.

    Attributes:
        pooler (nn.Module): The pooling layer.
        query_model (nn.Module): The query model.
        doc_model (nn.Module): The document model.
        q (torch.Tensor): The input IDs for the query.
        q_att (torch.Tensor): The attention mask for the query.
        scoring_function (callable): The scoring function to compute the similarity score.
    """

    def __init__(self, query_model, doc_model, pooler, query_input_ids, query_attention_mask,
                 scoring_function):
        super(DocModel, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model
        self.q = query_input_ids
        self.q_att = query_attention_mask
        self.scoring_function = scoring_function

    def forward_with_features(self, features, model):
        """
        Forward pass with additional features.

        Args:
            features (dict): Dictionary containing the input features.
            model (nn.Module): The model to forward the features through.

        Returns:
            torch.Tensor: The sentence embedding.
        """
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

    def forward(self, doc_input_ids , doc_attention_mask):
        """
        Forward pass of the document model.

        Args:
            doc_input_ids (torch.Tensor): The input IDs for the document.
            doc_attention_mask (torch.Tensor): The attention mask for the document.

        Returns:
            torch.Tensor: The diagonal of the similarity score matrix.
        """
        features_query = {'input_ids': self.q, 'attention_mask': self.q_att}
        features_doc = {'input_ids': doc_input_ids, 'attention_mask' : doc_attention_mask}

        q_emb = self.forward_with_features(features_query, self.query_model)
        doc_emb = self.forward_with_features(features_doc, self.doc_model)
        score = self.scoring_function(q_emb, doc_emb)
        return score.diagonal()