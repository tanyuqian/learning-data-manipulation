import os
import re
from io import open # pylint: disable=redefined-builtin
import tensorflow as tf
import texar as tx


def clean_sst_text(text):
    """Cleans tokens in the SST data, which has already been tokenized.
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


def transform_raw_sst(data_path, raw_fn, new_fn):
    """Transforms the raw data format to a new format.
    """
    fout_x_name = os.path.join(data_path, new_fn + '.sentences.txt')
    fout_x = open(fout_x_name, 'w', encoding='utf-8')
    fout_y_name = os.path.join(data_path, new_fn + '.labels.txt')
    fout_y = open(fout_y_name, 'w', encoding='utf-8')

    fin_name = os.path.join(data_path, raw_fn)
    with open(fin_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            parts = line.strip().split()
            label = parts[0]
            sent = ' '.join(parts[1:])
            sent = clean_sst_text(sent)
            fout_x.write(sent + '\n')
            fout_y.write(label + '\n')

    return fout_x_name, fout_y_name


def prepare_data(data_path):
    """Preprocesses SST2 data.
    """
    train_path = os.path.join(data_path, "sst.train.sentences.txt")
    if not tf.gfile.Exists(train_path):
        url = ('https://raw.githubusercontent.com/ZhitingHu/'
               'logicnn/master/data/raw/')
        files = ['stsa.binary.phrases.train', 'stsa.binary.dev',
                 'stsa.binary.test']
        for fn in files:
            tx.data.maybe_download(url + fn, data_path, extract=True)

    fn_train, _ = transform_raw_sst(
        data_path, 'stsa.binary.phrases.train', 'sst2.train')
    transform_raw_sst(data_path, 'stsa.binary.dev', 'sst2.dev')
    transform_raw_sst(data_path, 'stsa.binary.test', 'sst2.test')

    vocab = tx.data.make_vocab(fn_train)
    fn_vocab = os.path.join(data_path, 'sst2.vocab')
    with open(fn_vocab, 'w', encoding='utf-8') as f_vocab:
        for v in vocab:
            f_vocab.write(v + '\n')

    tf.logging.info('Preprocessing done: {}'.format(data_path))
