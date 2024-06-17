from torch import nn 

class QueryModel(nn.Module):
    """
    A wrapper class for performing information retrieval using a query and document models.

    Args:
        query_model (nn.Module): The query model.
        doc_model (nn.Module): The document model.
        pooler (nn.Module): The pooling layer for extracting sentence embeddings.
        doc_input_ids (torch.Tensor): The input IDs of the document.
        doc_attention_mask (torch.Tensor): The attention mask of the document.
        scoring_function (callable): The scoring function for computing the similarity score between the document and query embeddings.

    Attributes:
        pooler (nn.Module): The pooling layer for extracting sentence embeddings.
        query_model (nn.Module): The query model.
        doc_model (nn.Module): The document model.
        d (torch.Tensor): The input IDs of the document.
        d_att (torch.Tensor): The attention mask of the document.
        scoring_function (callable): The scoring function for computing the similarity score between the document and query embeddings.
    """

    def __init__(self, query_model, doc_model, pooler, doc_input_ids, doc_attention_mask,
                 scoring_function):
        super(QueryModel, self).__init__()
        self.pooler = pooler
        self.query_model = query_model
        self.doc_model = doc_model
        self.d = doc_input_ids
        self.d_att = doc_attention_mask
        self.scoring_function = scoring_function

    def _forward_with_features(self, features, model):
        """
        Forward pass with additional features.

        Args:
            features (dict): Dictionary containing the input features.
            model (nn.Module): The model to forward the features through.

        Returns:
            torch.Tensor: The sentence embedding obtained from the pooling layer.
        """
        trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        trans_features.update({"token_embeddings": output_tokens, "attention_mask": features["attention_mask"]})

        if model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            trans_features.update({"all_layer_embeddings": hidden_states})

        return self.pooler(trans_features)['sentence_embedding']

    def forward(self, query_input_ids, query_attention_mask):
        """
        Forward pass of the IRWrapperQuery module.

        Args:
            query_input_ids (torch.Tensor): The input IDs of the query.
            query_attention_mask (torch.Tensor): The attention mask of the query.

        Returns:
            torch.Tensor: The similarity score between the document and query embeddings.
        """
        features_doc = {'input_ids': self.d, 'attention_mask': self.d_att}
        features_query = {'input_ids': query_input_ids, 'attention_mask': query_attention_mask}

        q_emb = self._forward_with_features(features_query, self.query_model)
        doc_emb = self._forward_with_features(features_doc, self.doc_model)
        score = self.scoring_function(doc_emb, q_emb)

        return score.diagonal()