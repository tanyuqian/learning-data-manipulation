from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import random


BERT_MODEL = 'bert-base-uncased'
MAX_SEQ_LENGTH = 64


class Generator:
    def __init__(self, label_list, device):
        self._label_list = label_list

        self._tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL, do_lower_case=True)

        self._model = BertForMaskedLM.from_pretrained(BERT_MODEL)
        if len(self._label_list) != 2:
            self._model.bert.embeddings.token_type_embeddings = \
                nn.Embedding(len(label_list), 768)
            self._model.bert.embeddings.token_type_embeddings.weight.data.\
                normal_(mean=0.0, std=0.02)

        self._device = device
        self._model.to(self._device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def get_optimizer(self, learning_rate):
        self._optimizer = _get_optimizer(
            self._model, learning_rate=learning_rate)

    def load_data(self, set_type, examples, batch_size, shuffle):
        self._dataset[set_type] = RandomMaskedBERTDataset(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            max_seq_length=MAX_SEQ_LENGTH)

        self._data_loader[set_type] = DataLoader(
            self._dataset[set_type], batch_size=batch_size, shuffle=shuffle)

    def dev_loss(self):
        self._model.eval()
        sum_loss = 0.
        for step, batch in enumerate(self._data_loader['dev']):
            batch = tuple(t.to(self._device) for t in batch)
            _, input_ids, input_mask, segment_ids, masked_ids = batch

            loss = self._model(input_ids, segment_ids, input_mask, masked_ids)
            sum_loss += loss.item()

        return sum_loss

    def train_epoch(self):
        self._model.train()
        for step, batch in enumerate(self._data_loader['train']):
            batch = tuple(t.to(self._device) for t in batch)
            _, input_ids, input_mask, segment_ids, masked_ids = batch

            self._model.zero_grad()
            loss = self._model(input_ids, segment_ids, input_mask, masked_ids)
            loss.backward()
            self._optimizer.step()

        return self.dev_loss()

    def _augment_example(self, example, num_aug):
        features = _convert_example_to_features(
            example=example,
            label_list=self._label_list,
            max_seq_length=MAX_SEQ_LENGTH,
            tokenizer=self._tokenizer)

        init_ids, _, input_mask, segment_ids, _ = \
            (t.view(1, -1).to(self._device) for t in features)

        len = int(torch.sum(input_mask).item())
        if len >= 4:
            mask_idx = sorted(
                random.sample(list(range(1, len - 1)), max(len // 7, 2)))
        else:
            mask_idx = [1]

        masked_ids = init_ids[0][mask_idx]
        init_ids[0][mask_idx] = \
            self._tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        logits = self._model(init_ids, segment_ids, input_mask)[0]

        # Get 2 samples
        aug_probs_all = []
        for _ in range(num_aug):
            probs = F.gumbel_softmax(logits, hard=False) # TODO
            aug_probs = torch.zeros_like(probs).scatter_(
                1, init_ids[0].unsqueeze(1), 1.)
            for t in mask_idx:
                aug_probs = torch.cat(
                    [aug_probs[:t], probs[t:t + 1], aug_probs[t + 1:]], dim=0)

            aug_probs_all.append(aug_probs)

        aug_probs = torch.cat([ap.unsqueeze(0) for ap in aug_probs_all], dim=0)

        return aug_probs

    def _finetune_example(self, classifier, example,
                          finetune_batch_size, num_aug):
        aug_probs = self._augment_example(example, num_aug=num_aug)
        classifier.finetune_generator(
            example, aug_probs, finetune_batch_size)

    def finetune_batch(self, classifier, examples, num_aug=1):
        self._model.train()
        self._model.zero_grad()

        for example in examples:
            aug_probs = self._augment_example(example, num_aug=num_aug)
            classifier.finetune_generator(
                example, aug_probs, finetune_batch_size=len(examples))
        self._optimizer.step()

    def augment_batch(self, examples, num_aug=1):
        self._model.eval()

        aug_examples = []
        for example in examples:
            with torch.no_grad():
                aug_probs = self._augment_example(example, num_aug=num_aug)
                aug_examples.append((example, aug_probs))

        return aug_examples

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer


class RandomMaskedBERTDataset(Dataset):
    def __init__(self, examples, label_list, tokenizer, max_seq_length):
        self._examples = examples
        self._label_list = label_list
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, index):
        # generate different random masks every time.
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
    """
    this function is copied from
    https://github.com/IIEKES/cbert_aug/blob/master/aug_dataset_wo_ft.py#L119
    """

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    masked_lm_prob = 0.15
    max_predictions_per_seq = 20

    tokens_a = tokenizer.tokenize(example.text_a)
    segment_id = label_map[example.label]
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # 由于是CMLM，所以需要用标签
    tokens = []
    segment_ids = []
    # is [CLS]和[SEP] needed ？
    tokens.append("[CLS]")
    segment_ids.append(segment_id)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(segment_id)
    tokens.append("[SEP]")
    segment_ids.append(segment_id)
    masked_lm_labels = [-1] * max_seq_length

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    random.shuffle(cand_indexes)
    len_cand = len(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms_pos = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms_pos) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = tokens[cand_indexes[
                    random.randint(0, len_cand - 1)]]

        masked_lm_labels[index] = \
            tokenizer.convert_tokens_to_ids([tokens[index]])[0]
        output_tokens[index] = masked_token
        masked_lms_pos.append(index)

    init_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(output_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        init_ids.append(0)
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)  # ?segment_id

    assert len(init_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (torch.tensor(init_ids),
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(masked_lm_labels))


def _rev_wordpiece(str):
    if len(str) > 1:
        for i in range(len(str)-1, 0, -1):
            if str[i] == '[PAD]':
                str.remove(str[i])
            elif len(str[i]) > 1 and str[i][0] == '#' and str[i][1] == '#':
                str[i-1] += str[i][2:]
                str.remove(str[i])
    return " ".join(str[1:-1])