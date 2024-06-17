from sentence_transformers import SentenceTransformer, models

def load_sbert(model_name_or_path, pooling=None, max_seq_length=None):
    model = SentenceTransformer(model_name_or_path, cache_folder = "models")

    ## Check whether SBERT can load the checkpoint and use it
    ## Loadable by SBERT directly
    ## Mainly two cases: (1) The checkpoint is in SBERT-format (e.g. "bert-base-nli-mean-tokens"); (2) it is in HF-format but the last layer can provide a hidden state for each token (e.g. "bert-base-uncased")
    ## NOTICE: Even for (2), there might be some checkpoints (e.g. "princeton-nlp/sup-simcse-bert-base-uncased") that uses a linear layer on top of the CLS token embedding to get the final dense representation. In this case, setting `--pooling` to a specify pooling method will misuse the checkpoint. This is why we recommend to use SBERT-format if possible
    ## Setting pooling if needed
    if pooling is not None:
        last_layer: models.Pooling = model[-1]
        assert (
            type(last_layer) == models.Pooling
        ), f"The last layer is not a pooling layer and thus `--pooling={pooling}` cannot work. Please try leaving `--pooling=None` as in the default setting"
        # We here change the pooling by building the whole SBERT model again, which is safer and more maintainable than setting the attributes of the Pooling module
        word_embedding_model = models.Transformer(
            model_name_or_path, max_seq_length=max_seq_length
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=pooling,
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if max_seq_length is not None:
        first_layer: models.Transformer = model[0]
        assert (
            type(first_layer) == models.Transformer
        ), "Unknown error, please report this"
        assert hasattr(
            first_layer, "max_seq_length"
        ), "Unknown error, please report this"
        setattr(
            first_layer, "max_seq_length", max_seq_length
        )  # Set the maximum-sequence length

    return model