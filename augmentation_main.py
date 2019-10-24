import argparse
import os
from tqdm import trange
import time
import random

import torch

from data_utils.data_processors import get_data

from augmentation.classifier import Classifier
from augmentation.generator import Generator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

parser = argparse.ArgumentParser()

# data
parser.add_argument("--task", default="sst-5", type=str)
parser.add_argument('--train_num_per_class', default=None, type=int)
parser.add_argument('--dev_num_per_class', default=None, type=int)
parser.add_argument('--data_seed', default=159, type=int)

# train
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--classifier_lr", default=4e-5, type=float)
parser.add_argument("--classifier_pretrain_epochs", default=1, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--min_epochs", default=0, type=int)

# augmentation
parser.add_argument("--generator_lr", default=4e-5, type=float)
parser.add_argument("--generator_pretrain_epochs", default=60, type=int)
parser.add_argument('--n_aug', default=2, type=int)

args = parser.parse_args()
print(args)

random.seed(args.data_seed)


def _pretrain_classifier(classifier, train_examples):
    print('=' * 60, '\n', 'Classifier Pre-training', '\n', '=' * 60, sep='')
    for epoch in range(args.classifier_pretrain_epochs):
        train_epoch(
            epoch=-1,
            generator=None,
            classifier=classifier,
            train_examples=train_examples,
            do_augment=False)
        dev_acc = classifier.evaluate('dev')

        print('Classifier pretrain Epoch {}, Dev Acc: {:.4f}'.format(
            epoch, 100. * dev_acc))


def _pretrain_generator(generator, train_examples):
    print('=' * 60, '\n', 'Generator Pre-training', '\n', '=' * 60, sep='')
    best_dev_loss = 1e10
    tag = time.time()

    for epoch in range(args.generator_pretrain_epochs):
        dev_loss = generator.train_epoch()

        if dev_loss < best_dev_loss:
            torch.save(generator.model.state_dict(),
                       '/tmp/generator_model_{}.pt'.format(tag))
            torch.save(generator.optimizer.state_dict(),
                       '/tmp/generator_optimizer_{}.pt'.format(tag))
            best_dev_loss = dev_loss

        print('Epoch {}, Dev Loss: {:.4f}; Best Dev Loss: {:.4f}'.format(
            epoch, dev_loss, best_dev_loss))

    generator.model.load_state_dict(
        torch.load('/tmp/generator_model_{}.pt'.format(tag)))
    generator.optimizer.load_state_dict(
        torch.load('/tmp/generator_optimizer_{}.pt'.format(tag)))

    os.remove(os.path.join('/tmp', 'generator_model_{}.pt'.format(tag)))
    os.remove(os.path.join('/tmp', 'generator_optimizer_{}.pt'.format(tag)))


def train_epoch(epoch, generator, classifier, train_examples, do_augment):
    random.seed(199 * (epoch + 1))
    random.shuffle(train_examples)

    batch_size = args.batch_size
    for i in trange(0, len(train_examples), batch_size, desc='Training'):
        batch_examples = train_examples[i: i + batch_size]
        if do_augment:
            generator.finetune_batch(
                classifier=classifier,
                examples=batch_examples,
                num_aug=args.n_aug)

            batch_examples = generator.augment_batch(
                examples=batch_examples, num_aug=1)

        classifier.train_batch(batch_examples, is_augment=do_augment)


def main():
    examples, label_list = get_data(
        task=args.task,
        train_num_per_class=args.train_num_per_class,
        dev_num_per_class=args.dev_num_per_class,
        data_seed=args.data_seed)

    generator = Generator(label_list=label_list, device=device)
    generator.get_optimizer(learning_rate=args.generator_lr)
    generator.load_data('train', examples['train'],
                        batch_size=args.batch_size, shuffle=True)
    generator.load_data('dev', examples['dev'],
                        batch_size=args.batch_size, shuffle=False)

    classifier = Classifier(label_list=label_list, device=device)
    classifier.get_optimizer(learning_rate=args.classifier_lr)
    classifier.load_data('dev', examples['dev'], batch_size=20, shuffle=True)
    classifier.load_data('test', examples['test'], batch_size=20, shuffle=True)

    _pretrain_classifier(classifier, examples['train'])
    _pretrain_generator(generator, examples['train'])

    print('=' * 60, '\n', 'Training', '\n', '=' * 60, sep='')
    best_dev_acc, final_test_acc = -1., -1.
    for epoch in range(args.epochs):
        train_epoch(
            epoch=epoch,
            generator=generator,
            classifier=classifier,
            train_examples=examples['train'],
            do_augment=True)
        dev_acc = classifier.evaluate('dev')

        if epoch >= args.min_epochs:
            do_test = (dev_acc > best_dev_acc)
            best_dev_acc = max(best_dev_acc, dev_acc)
        else:
            do_test = False

        print('Epoch {}, Dev Acc: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, 100. * dev_acc, 100. * best_dev_acc))

        if do_test:
            final_test_acc = classifier.evaluate('test')
            print('Test Acc: {:.4f}'.format(100. * final_test_acc))

    print('Final Dev Acc: {:.4f}, Final Test Acc: {:.4f}'.format(
        100. * best_dev_acc, 100. * final_test_acc))


if __name__ == '__main__':
    main()
