from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.modeling import BertEncoder, BertPooler, BertLayerNorm

import torch
from torch import nn
from torch.nn import functional as F


class SoftEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))

    def forward(self, ids_or_probs, use_probs=False):
        if not use_probs:
            ids = ids_or_probs
            assert len(ids.shape) == 2
            probs = torch.zeros(
                ids.shape[0], ids.shape[1], self.num_embeddings,
                device=ids_or_probs.device).scatter_(2, ids.unsqueeze(2), 1.)
        else:
            probs = ids_or_probs

        embedding = probs.view(-1, self.num_embeddings).mm(self.weight).\
            view(probs.shape[0], probs.shape[1], self.embedding_dim)

        return embedding


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = SoftEmbedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = SoftEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SoftEmbedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids_or_probs, token_type_ids=None,
                use_input_probs=False):
        seq_length = input_ids_or_probs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids_or_probs.device)
        position_ids = position_ids.unsqueeze(0).\
            expand(input_ids_or_probs.shape[:2])
        assert token_type_ids is not None
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = \
            self.word_embeddings(input_ids_or_probs, use_probs=use_input_probs)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids_or_probs, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, use_input_probs=False):
        assert attention_mask is not None
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = \
            extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = \
            self.embeddings(input_ids_or_probs, token_type_ids, use_input_probs)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids_or_probs, token_type_ids=None,
                attention_mask=None, labels=None, use_input_probs=False):
        _, pooled_output = self.bert(
            input_ids_or_probs, token_type_ids, attention_mask,
            output_all_encoded_layers=False, use_input_probs=use_input_probs)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits