from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from tqdm import tqdm
import math

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader, TensorDataset
from data_utils.data_processors import InputFeatures


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64


class Classifier:
    def __init__(self, label_list, device):
        self._label_list = label_list
        self._device = device

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=len(label_list)).to(device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = _make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch

            self._optimizer.zero_grad()
            loss = self._model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()
            self._optimizer.step()

    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]

        for batch in tqdm(data_loader,
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch[:4]

            with torch.no_grad():
                logits = self._model(input_ids, segment_ids, input_mask)
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds)
            labels_all.append(label_ids)

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def _get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)


def _make_data_loader(examples, label_list, tokenizer, batch_size, shuffle):
    all_features = _convert_examples_to_features(
        examples=examples,
        label_list=label_list,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        output_mode='classification')

    all_input_ids = torch.tensor(
        [f.input_ids for f in all_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in all_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in all_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in all_features], dtype=torch.long)
    all_ids = torch.arange(len(examples))

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then each
    # token that's truncated likely contains more information than a longer
    # sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
