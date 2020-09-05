from tqdm import tqdm
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from torchvision import models

from magic_module import MagicModule


EPSILON = 1e-5


class ImageClassifier:
    def __init__(self, norm_fn='linear', pretrained=True,
                 baseline=False, ren=False, softmax_temp=1.):
        if ren or baseline:
            assert norm_fn == 'linear'

        self._model = models.resnet34(pretrained=pretrained).to('cuda')

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

        self._weights = None
        self._w_decay = None

        self._baseline = baseline
        self._ren = ren

        self._norm_fn = norm_fn
        self._softmax_temp = softmax_temp

    def init_weights(self, n_examples, w_init, w_decay):
        assert self._ren is False and self._baseline is False
        self._weights = torch.tensor(
            [w_init] * n_examples, requires_grad=True).to('cuda')
        self._w_decay = w_decay

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = examples

        all_inputs = torch.tensor([t.input.tolist() for t in examples])
        all_labels = torch.tensor([t.label for t in examples])
        all_ids = torch.arange(len(examples))

        self._data_loader[set_type] = DataLoader(
            TensorDataset(all_inputs, all_labels, all_ids),
            batch_size=batch_size, shuffle=shuffle)

    def get_optimizer(self, learning_rate, momentum, weight_decay):
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def pretrain_epoch(self):
        self.train_epoch(is_pretrain=True)

    def train_epoch(self, is_pretrain=False):
        criterion = nn.CrossEntropyLoss(reduction='none')
        for batch in tqdm(self._data_loader['train'], desc='Training Epoch'):
            inputs, labels, ids = tuple(t.to('cuda') for t in batch)

            if is_pretrain:
                weights = linear_normalize(
                    torch.ones(inputs.shape[0]).to('cuda'))
            else:
                if self._norm_fn == 'softmax':
                    weights = softmax_normalize(
                        self._get_weights(batch),
                        temperature=self._softmax_temp)
                else:
                    weights = linear_normalize(self._get_weights(batch))

            self._model.train()

            self._optimizer.zero_grad()
            logits = self._model(inputs)
            loss = criterion(logits, labels)
            loss = torch.sum(loss * weights.data)
            loss.backward()
            self._optimizer.step()

    def _get_weights(self, batch):
        self._model.eval()

        inputs, labels, ids = tuple(t.to('cuda') for t in batch)
        batch_size = inputs.shape[0]

        if self._baseline:
            return torch.ones(batch_size).to('cuda')
        elif self._ren:
            weights = torch.zeros(batch_size, requires_grad=True).to('cuda')
        else:
            weights = self._weights[ids]

        magic_model = MagicModule(self._model)
        criterion = nn.CrossEntropyLoss()

        model_tmp = copy.deepcopy(self._model)
        optimizer_hparams = self._optimizer.state_dict()['param_groups'][0]
        optimizer_tmp = optim.SGD(
            model_tmp.parameters(),
            lr=optimizer_hparams['lr'],
            momentum=optimizer_hparams['momentum'],
            weight_decay=optimizer_hparams['weight_decay'])

        for i in range(batch_size):
            model_tmp.load_state_dict(self._model.state_dict())
            optimizer_tmp.load_state_dict(self._optimizer.state_dict())

            model_tmp.zero_grad()

            if i > 0:
                l, r, t = i - 1, i + 1, 1
            else:
                l, r, t = i, i + 2, 0

            logits = model_tmp(inputs[l:r])[t:t+1]
            loss = criterion(logits, labels[i:i+1])
            loss.backward()
            optimizer_tmp.step()

            deltas = {}
            for (name, param), (name_tmp, param_tmp) in zip(
                    self._model.named_parameters(),
                    model_tmp.named_parameters()):
                assert name == name_tmp
                deltas[name] = weights[i] * (param_tmp.data - param.data)
            magic_model.update_params(deltas)

        weights_grad_list = []
        for step, val_batch in enumerate(self._data_loader['dev']):
            val_batch = (t.to('cuda') for t in val_batch)
            val_inputs, val_labels, _ = val_batch
            val_batch_size = val_labels.shape[0]

            if weights.grad is not None:
                weights.grad.zero_()
            val_logits = magic_model(val_inputs)
            val_loss = criterion(val_logits, val_labels)
            val_loss = val_loss * float(val_batch_size) / float(
                len(self._dataset['dev']))

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
        for batch in tqdm(self._data_loader[set_type],
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to('cuda') for t in batch)
            inputs, labels, _ = batch

            with torch.no_grad():
                logits = self._model(inputs)

            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds)
            labels_all.append(labels)

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)
