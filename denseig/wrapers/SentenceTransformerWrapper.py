from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers.util import dot_score
from beir.retrieval import models
import torch


class SentenceTransformerWrapper:
    """
    A wrapper class for SentenceTransformer model.

    Args:
        model (SentenceTransformer): The SentenceTransformer model.
        device (str): The device to run the model on.

    Attributes:
        model (SentenceTransformer): The SentenceTransformer model.
        device (str): The device to run the model on.
        bert_model (DistilBertModel): The DistilBertModel from huggingface.
        pooler (nn.Module): The pooler module of the model.
        tokenizer (Tokenizer): The tokenizer used by the model.
        bert_tokenizer (Callable): A wrapper function that tokenizes inputs.
        ref_token_id (int): The token used for generating token reference.
        sep_token_id (int): The token used for adding it to the end of the text.
        cls_token_id (int): The token used for adding it to the beginning of the text.
        searcher (DenseRetrievalExactSearch): The DenseRetrievalExactSearch object.

    Methods:
        _from_sbert_to_beir: Converts the SentenceTransformer model to BEIR format.
        return_text_and_base_features: Returns the encoded text and base features.
        decode: Decodes the input_ids into text.
        _produce_embedding: Produces the embedding for the input text.
        calculate_sim: Calculates the similarity between two texts.
        return_top_k: Returns the top k documents based on the query.
        return_hard_negatives: Returns the hard negatives for the query.
        return_attention: Returns the attention weights for the input text.
    """

    def __init__(self, model, device):
        self.model = model 
        self.device = device
        self.bert_model = model._first_module().auto_model
        self.bert_model.to(device)
        self.bert_model.eval()
        self.bert_model.zero_grad()
        self.pooler =  model._last_module()
        self.tokenizer = model._first_module().tokenizer 
        self.bert_tokenizer = model._first_module().tokenize
        self.ref_token_id = self.tokenizer.pad_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.searcher = DenseRetrievalExactSearch(self._from_sbert_to_beir(), batch_size=256, corpus_chunk_size=256*1000) 
    
    def _from_sbert_to_beir(self):
        """
        Converts the SentenceTransformer model to BEIR format.

        Returns:
            SentenceBERT: The converted SentenceBERT model.
        """
        retriever = models.SentenceBERT(sep=" ")
        retriever.q_model = self.model
        retriever.doc_model = self.model
        return retriever
    
    def return_text_and_base_features(self, text):
        """
        Returns the encoded text and base features.

        Args:
            text (str): The input text.

        Returns:
            tuple: A tuple containing the encoded text, fake input, and tokenized input.
        """
        q_encoded = self.bert_tokenizer(text)
        q_fake = [self.cls_token_id] + [self.ref_token_id] * (q_encoded['attention_mask'].shape[1] - 2) + [self.sep_token_id]
        fake = {"input_ids": torch.tensor([q_fake], device=self.device), "attention_mask": torch.clone(q_encoded['attention_mask']).to(self.device)}
        q_encoded['input_ids'] = q_encoded['input_ids'].to(self.device)
        q_encoded['attention_mask'] = q_encoded['attention_mask'].to(self.device)
        return q_encoded, fake, self.tokenizer.convert_ids_to_tokens(q_encoded['input_ids'][0])

    def decode(self, input_ids):
        """
        Decodes the input_ids into text.

        Args:
            input_ids (torch.Tensor): The input_ids to decode.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(token_ids=input_ids[0])

    def _produce_embedding(self, input_text):
        """
        Produces the embedding for the input text.

        Args:
            input_text (str): The input text.

        Returns:
            torch.Tensor: The embedding of the input text.
        """
        input_texts = [input_text]
        return self.model.encode(input_texts)[0]

    def calculate_sim(self, query, doc):
        """
        Calculates the similarity between two texts.

        Args:
            query (str): The query text.
            doc (str): The document text.

        Returns:
            float: The similarity score between the query and document.
        """
        return dot_score(self._produce_embedding(query), self._produce_embedding(doc))
    

    
    def return_attention(self, text):
        """
        Returns the attention weights for the input text.

        Args:
            text (str): The input text.

        Returns:
            tuple: A tuple containing the attention weights and tokens.
        """
        inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        outputs = self.bert_model(inputs, output_attentions=True)
        attention = outputs[-1]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs[0]) 
        return attention, tokens
