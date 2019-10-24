from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

from tqdm import tqdm
import math

import torch
from torch import nn
from torch import optim

from torch.utils.data import DataLoader, TensorDataset
from data_utils.data_processors import InputFeatures

from magic_module import MagicModule


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64
EPSILON = 1e-5


class Classifier:
    def __init__(self, label_list, ren, norm_fn, device):
        self._label_list = label_list
        self._ren = ren
        self._device = device

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=len(label_list)).to(device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        if norm_fn == 'linear':
            self._norm_fn = _linear_normalize
        elif norm_fn == 'softmax':
            self._norm_fn = _softmax_normalize

        if ren:
            assert norm_fn == 'linear'

    def init_weights(self, n_examples, w_init, w_decay):
        if self._ren:
            raise ValueError(
                'no global weighting initialization when \'ren\'=True')

        self._weights = torch.tensor(
            [w_init] * n_examples, requires_grad=True).to(device=self._device)
        self._w_decay = w_decay

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

    def pretrain_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Pre-training')):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch

            self._optimizer.zero_grad()
            loss = self._model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()
            self._optimizer.step()

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, _ = batch

            batch_size = batch[-1].shape[0]
            weights = []
            for i in range(0, batch_size, 8):
                lil_batch = tuple(t[i:i+8] for t in batch)
                weights.append(self._get_weights(lil_batch))
            weights = self._norm_fn(torch.cat(weights, dim=0))

            self._optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss(reduction='none')
            logits = self._model(input_ids, segment_ids, input_mask)
            loss = criterion(logits, label_ids)
            loss = torch.sum(loss * weights.data)
            loss.backward()
            self._optimizer.step()

    def _get_weights(self, batch):
        input_ids, input_mask, segment_ids, label_ids, ids = batch
        batch_size = label_ids.shape[0]

        optimizer_initialized = ('exp_avg' in self._optimizer.state[
            next(self._model.parameters())])
        if not optimizer_initialized:
            return torch.ones(batch_size).to(self._device)

        if self._ren:
            weights = torch.zeros(
                batch_size, requires_grad=True).to(self._device)
        else:
            weights = self._weights[ids]

        magic_model = MagicModule(self._model)
        criterion = nn.CrossEntropyLoss()

        for i in range(batch_size):
            self._model.zero_grad()
            logits = self._model(
                input_ids[i:i + 1], segment_ids[i:i + 1], input_mask[i:i + 1])
            loss = criterion(logits, label_ids[i:i + 1])

            grads = torch.autograd.grad(
                loss, [param for name, param in self._model.named_parameters()])
            grads = {param: grads[j] for j, (name, param) in enumerate(
                self._model.named_parameters())}

            deltas = _adam_delta(self._optimizer, self._model, grads)
            deltas = {name: weights[i] * delta.data for name, delta in
                      deltas.items()}
            magic_model.update_params(deltas)

        weights_grad_list = []
        for step, val_batch in enumerate(self._data_loader['dev']):
            val_batch = (t.to(self._device) for t in val_batch)
            val_input_ids, val_input_mask, val_segment_ids, val_label_ids, _ = \
                val_batch
            val_batch_size = val_label_ids.shape[0]

            val_loss = magic_model(
                val_input_ids, val_segment_ids, val_input_mask, val_label_ids)
            val_loss = val_loss * \
                       float(val_batch_size) / float(len(self._dataset['dev']))

            weights_grad = torch.autograd.grad(
                val_loss, weights, retain_graph=True)[0]
            weights_grad_list.append(weights_grad)

        weights_grad = sum(weights_grad_list)

        if self._ren:
            return -weights_grad
        else:
            self._weights[ids] = weights.data / self._w_decay - weights_grad
            self._weights[ids] = torch.max(self._weights[ids], torch.ones_like(
                self._weights[ids]).fill_(EPSILON))

            return self._weights[ids].data

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


def _linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def _softmax_normalize(weights):
    return nn.functional.softmax(weights, dim=0)


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


def _adam_delta(optimizer, model, grads):
    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            step = state['step'] + 1

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            bias_correction1 = 1. - beta1 ** step
            bias_correction2 = 1. - beta2 ** step
            step_size = group['lr'] * math.sqrt(
                bias_correction2) / bias_correction1

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}
