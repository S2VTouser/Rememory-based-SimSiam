from __future__ import division
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils


def split_dataset_by_labels(X, y, task_labels, nb_classes=None, multihead=False):
    """Split dataset by labels.

    Args:
        X: data
        y: labels
        task_labels: list of list of labels, one for each dataset
        nb_classes: number of classes (used to convert to one-hot)
    Returns:
        List of (X, y) tuples representing each dataset
    """
    if nb_classes is None:
        nb_classes = len(np.unique(y))
    datasets = []
    for labels in task_labels:
        idx = np.in1d(y, labels)  # 返回长度为y的布尔数组 判断y中的每个元素是否在labels中
        if multihead:
            label_map = np.arange(nb_classes)
            label_map[labels] = np.arange(len(labels))
            data = X[idx], np_utils.to_categorical(label_map[y[idx]], len(labels))
        else:
            data = X[idx], np_utils.to_categorical(y[idx], nb_classes)
        datasets.append(data)
    return datasets


def split_dataset_randomly(X, y, nb_splits, nb_classes=None):
    """Split dataset by labels.

    Args:
        X: data
        y: labels
        nb_splits: number of splits to return
        task_labels: list of list of labels, one for each dataset
        nb_classes: number of classes (used to convert to one-hot)
    Returns:
        List of (X, y) tuples representing each dataset
    """
    if nb_classes is None:
        nb_classes = len(np.unique(y))
    datasets = []
    idx = range(len(y))
    np.random.shuffle(idx)
    print('idx:',idx)
    split_size = len(y) // nb_splits
    for i in range(nb_splits):
        data = X[idx[split_size * i:split_size * (i + 1)]], np_utils.to_categorical(
            y[idx[split_size * i:split_size * (i + 1)]], nb_classes)
        datasets.append(data)
        print('data:',data)
    return datasets


def construct_split_cifar10(nb_splits=5, split='train'):
    """Split CIFAR10 dataset by labels.

        Args:
            task_labels: list of list of labels, one for each dataset
            split: whether to use train or testing data

        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load CIFAR10 data and normalize
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # X_train = X_train.reshape(-1, 3, 32, 32)
    # X_test = X_test.reshape(-1, 32**2)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    no = X_train.max() #normalize
    print('no:', no)
    X_train /= no
    X_test /= no

    if split == 'train':
        X, y = X_train, y_train
    else:
        X, y = X_test, y_test

    # return split_dataset_by_labels(X, y, task_labels, nb_classes)
    return split_dataset_randomly(X, y, nb_splits, nb_classes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    random.seed(1)

    ds = construct_split_cifar10()
    print('ds:',ds)
    plt.subplot(121)
    plt.imshow(ds[0][0][0].transpose((1, 2, 0)), interpolation='nearest')
    plt.subplot(122)
    plt.imshow(ds[1][0][0].transpose((1, 2, 0)), interpolation='nearest')
    plt.show()
