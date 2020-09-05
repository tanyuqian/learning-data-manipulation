import random

import torchvision
from torchvision import transforms


def get_data(train_num_per_class, dev_num_per_class, imbalance_rate=1.,
             data_seed=159):
    random.seed(data_seed)
    processor = CIFAR10Processor()

    # if running imbalance setting, keep only 2 labels.
    if imbalance_rate < 1.:
        processor.set_labels(list(range(2)))
    else:
        processor.set_labels(list(range(10)))

    examples = {}

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

    for split, examples_split in examples.items():
        print(f'#{split}: {len(examples_split)}')

    return examples, processor.get_labels()


class DataExample:
    def __init__(self, input, label):
        self.input = input
        self.label = label


class CIFAR10Processor:
    def __init__(self):
        self._labels = None

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def set_labels(self, labels):
        self._labels = labels

    def get_train_examples(self, num_per_class=None):
        train_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=self._transform)

        all_examples = [DataExample(input=t[0], label=t[1])
                        for t in train_set][:-500]

        return _subsample_by_classes(
            all_examples=all_examples, labels=self.get_labels(),
            num_per_class=num_per_class)

    def get_dev_examples(self, num_per_class=None):
        dev_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=self._transform)

        all_examples = [DataExample(input=t[0], label=t[1])
                        for t in dev_set][-500:]

        return _subsample_by_classes(
            all_examples=all_examples, labels=self.get_labels(),
            num_per_class=num_per_class)

    def get_test_examples(self):
        test_set = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=self._transform)

        return [DataExample(input=t[0], label=t[1])
                for t in test_set if t[1] in self.get_labels()]

    def get_labels(self):
        return self._labels


def _subsample_by_classes(all_examples, labels, num_per_class=None):
    if num_per_class is None:
        return all_examples

    examples = {label: [] for label in labels}
    for example in all_examples:
        if example.label in labels:
            examples[example.label].append(example)

    picked_examples = []
    for label in labels:
        random.shuffle(examples[label])

        examples_with_label = examples[label][:num_per_class[label]]
        picked_examples.extend(examples_with_label)

        print(f'number of examples with label \'{label}\': '
              f'{len(examples_with_label)}')

    return picked_examples
