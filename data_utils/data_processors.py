import random

import torchtext
from data_utils.download_sst2 import prepare_data


def _get_processor(task):
    if task == 'sst-5':
        return SST5Processor()
    elif task == 'sst-2':
        return SST2Processor()
    else:
        raise ValueError('Unknown task')


def get_data(task, train_num_per_class, dev_num_per_class, imbalance_rate=1., data_seed=159):
    random.seed(data_seed)
    processor = _get_processor(task)

    examples = dict()

    if train_num_per_class is not None:
        train_num_per_class = {
            label: train_num_per_class for label in processor.get_labels()}
        train_num_per_class[processor.get_labels()[0]] = int(
            train_num_per_class[processor.get_labels()[0]] * imbalance_rate)
    examples['train'] = processor.get_train_examples(train_num_per_class)

    if dev_num_per_class is not None:
        dev_num_per_class = {
            label: dev_num_per_class for label in processor.get_labels()}
    examples['dev'] = processor.get_dev_examples(dev_num_per_class)

    examples['test'] = processor.get_test_examples()

    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))

    return examples, processor.get_labels()


def _subsample_by_classes(all_examples, labels, num_per_class=None):
    if num_per_class is None:
        return all_examples

    examples = {label: [] for label in labels}
    for example in all_examples:
        examples[example.label].append(example)

    selected_examples = []
    for label in labels:
        random.shuffle(examples[label])

        num_in_class = num_per_class[label]
        selected_examples = selected_examples + examples[label][:num_in_class]
        print('number of examples with label \'{}\': {}'.format(
            label, num_in_class))

    return selected_examples


def _split_by_classes(all_examples, labels, num_select_per_class):
    examples = {label: [] for label in labels}
    for example in all_examples:
        examples[example.label].append(example)

    selected_examples = []
    remaining_examples = []
    for label in labels:
        assert num_select_per_class <= len(examples[label])

        random.shuffle(examples[label])
        selected_examples = \
            selected_examples + examples[label][:num_select_per_class]
        remaining_examples = \
            remaining_examples + examples[label][num_select_per_class:]

    return selected_examples, remaining_examples


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


class DatasetProcessor:
    def get_train_examples(self):
        raise NotImplementedError

    def get_dev_examples(self):
        raise NotImplementedError

    def get_test_examples(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError


class SST5Processor(DatasetProcessor):
    """Processor for the SST-5 data set."""

    def __init__(self):
        TEXT = torchtext.data.Field()
        LABEL = torchtext.data.Field(sequential=False)

        self._train_set, self._dev_set, self._test_set = \
            torchtext.datasets.SST.splits(
                TEXT, LABEL, fine_grained=True)

    def get_train_examples(self, num_per_class=None, noise_rate=0.):
        """See base class."""
        print('getting train examples...')
        all_examples = self._create_examples(self._train_set, "train")

        # Add noise
        for i, _ in enumerate(all_examples):
            if random.random() < noise_rate:
                all_examples[i].label = random.choice(self.get_labels())

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_dev_examples(self, num_per_class=None):
        """See base class."""
        print('getting dev examples...')
        all_examples = self._create_examples(self._dev_set, "dev")

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_test_examples(self):
        """See base class."""
        print('getting test examples...')
        return self._create_examples(self._test_set, "test")

    def get_labels(self):
        """See base class."""
        return ['negative', 'very positive', 'neutral',
                'positive', 'very negative']

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(
                guid=guid,
                text_a=' '.join(data.text),
                text_b=None,
                label=data.label))
        return examples


class SST2Processor(DatasetProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self):
        prepare_data('./data')

    def get_train_examples(self, num_per_class=None, noise_rate=0.):
        print('getting train examples...')
        all_examples = self._create_examples("train")

        # Add noise
        for i, _ in enumerate(all_examples):
            if random.random() < noise_rate:
                all_examples[i].label = random.choice(self.get_labels())

        return _subsample_by_classes(
            all_examples, self.get_labels(), num_per_class)

    def get_dev_examples(self, num_per_class=None):
        print('getting dev examples...')
        return _subsample_by_classes(
            self._create_examples("dev"), self.get_labels(), num_per_class)

    def get_test_examples(self):
        print('getting test examples...')
        return self._create_examples("test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, set_type):
        """Creates examples for the training and dev sets."""
        sentence_file = open('data/sst2.{}.sentences.txt'.format(set_type))
        labels_file = open('data/sst2.{}.labels.txt'.format(set_type))

        examples = []
        for sentence, label in zip(
                sentence_file.readlines(), labels_file.readlines()):
            label = label.strip('\n')
            sentence = sentence.strip('\n')

            if label == '':
                break
            examples.append(InputExample(
                guid=set_type, text_a=sentence, text_b=None, label=label))
        return examples
