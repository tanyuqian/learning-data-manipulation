from .text_data_processors import get_data as get_text_data
from .image_data_processors import get_data as get_image_data


def get_data(task, train_num_per_class, dev_num_per_class, imbalance_rate,
             data_seed):
    if task in ['sst-2', 'sst-5']:
        return get_text_data(
            task=task,
            train_num_per_class=train_num_per_class,
            dev_num_per_class=dev_num_per_class,
            imbalance_rate=imbalance_rate,
            data_seed=data_seed)

    elif task in ['cifar-10']:
        return get_image_data(
            train_num_per_class=train_num_per_class,
            dev_num_per_class=dev_num_per_class,
            imbalance_rate=imbalance_rate,
            data_seed=data_seed)