from pytorch_pretrained_bert.tokenization import BertTokenizer
from augmentation.bert_model import BertForSequenceClassification

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from magic_module import MagicModule
import math
from tqdm import tqdm


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64


class Classifier:
    def __init__(self, label_list, device):
        self._label_list = label_list

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)

        self._model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL, num_labels=len(self._label_list))

        self._device = device
        self._model.to(self._device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def get_optimizer(self, learning_rate):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = BERTDataset(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            max_seq_length=MAX_SEQ_LENGTH)

        self._data_loader[set_type] = DataLoader(
            self._dataset[set_type], batch_size=batch_size, shuffle=shuffle)

    def train_batch(self, train_examples, is_augment):
        features = []
        for example in train_examples:
            if is_augment:
                example = example[0]

            features.append(_convert_example_to_features(
                example=example,
                label_list=self._label_list,
                max_seq_length=MAX_SEQ_LENGTH,
                tokenizer=self._tokenizer))

        input_ids_or_probs, input_masks, segment_ids, label_ids = [torch.cat(
            [t[i].unsqueeze(0) for t in features], dim=0).to(
            self._device) for i in range(4)]
        if is_augment:
            num_aug = len(train_examples[0][1])

            input_ids_or_probs_aug = []
            for i in range(num_aug):
                for example in train_examples:
                    input_ids_or_probs_aug.append(example[1][i:i+1])
            input_ids_or_probs_aug = \
                torch.cat(input_ids_or_probs_aug, dim=0).to(self._device)

            inputs_onehot = torch.zeros_like(
                input_ids_or_probs_aug[:len(input_ids_or_probs)]).scatter_(
                2, input_ids_or_probs.unsqueeze(2), 1.)
            input_ids_or_probs = torch.cat(
                [inputs_onehot, input_ids_or_probs_aug], dim=0).to(self._device)

            segment_ids = \
                torch.cat([segment_ids] * (num_aug+1), dim=0).to(self._device)
            input_masks = \
                torch.cat([input_masks] * (num_aug+1), dim=0).to(self._device)
            label_ids = \
                torch.cat([label_ids] * (num_aug+1), dim=0).to(self._device)

        self._model.train()
        self._optimizer.zero_grad()
        loss = self._model(
            input_ids_or_probs, segment_ids, input_masks, label_ids,
            use_input_probs=is_augment)
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

    def finetune_generator(self, example, aug_probs, finetune_batch_size):
        magic_model = MagicModule(self._model)

        features = _convert_example_to_features(
            example=example,
            label_list=self._label_list,
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=self._tokenizer)

        _, input_mask, segment_ids, label_ids = \
            (t.to(self._device).unsqueeze(0) for t in features)

        num_aug = len(aug_probs)
        input_mask_aug = torch.cat([input_mask] * num_aug, dim=0)
        segment_ids_aug = torch.cat([segment_ids] * num_aug, dim=0)
        label_ids_aug = torch.cat([label_ids] * num_aug, dim=0)

        self._model.zero_grad()
        loss = self._model(
            aug_probs, segment_ids_aug, input_mask_aug, label_ids_aug,
            use_input_probs=True)
        grads = torch.autograd.grad(
            loss, [param for name, param in self._model.named_parameters()],
            create_graph=True)

        grads = {param: grads[i] for i, (name, param) in enumerate(
            self._model.named_parameters())}

        deltas = _adam_delta(self._optimizer, self._model, grads)
        magic_model.update_params(deltas)

        for step, batch in enumerate(self._data_loader['dev']):
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            dev_loss = magic_model(
                input_ids, segment_ids, input_mask, label_ids)
            dev_loss = dev_loss / len(self._data_loader['dev']) / \
                       finetune_batch_size / num_aug
            dev_loss.backward()

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer


class BERTDataset(Dataset):
    def __init__(self, examples, label_list, tokenizer, max_seq_length):
        self._examples = examples
        self._label_list = label_list
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, index):
        return _convert_example_to_features(
            example=self._examples[index],
            label_list=self._label_list,
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer)


def _get_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)


def _convert_example_to_features(
        example, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}

    tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:max_seq_length - 2]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    return (torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(label_id))


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