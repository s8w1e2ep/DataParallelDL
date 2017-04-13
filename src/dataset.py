import cPickle
import numpy as np
import os
import sys
import re
import cPickle
import tarfile
from six.moves import urllib

def open_mnist_dataset(start, length, pickle_file='/home/rche/data/notMNIST.pickle'):
    with open(pickle_file, 'rb') as f:
        save = cPickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        train_dataset, train_labels = reformat(train_dataset, train_labels)
        valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
        test_dataset, test_labels = reformat(test_dataset, test_labels)
    return train_dataset[start:start+length], train_labels[start:start+length], valid_dataset, valid_labels, test_dataset, test_labels

def reformat(dataset, labels):
    image_size = 28
    num_labels = 10
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def open_cifar10_dataset(start, length):
    data_dir = '/home/rche/data/cifar-10-batches-py'
    return load_and_preprocess_input(data_dir, start, length)

def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        try:
            d = cPickle.load(fp)
            return d
        except EOFError:
            pass

def prepare_input(data=None, labels=None):
    image_height = 32
    image_width = 32
    image_depth = 3
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    return data, labels

def load_and_preprocess_input(dataset_dir, start, length):
    r_data_file = re.compile('^data_batch_\d+')
    n_validate_samples = 2000
    n_test_samples = 5
    assert(os.path.isdir(dataset_dir))
    trn_all_data=[]
    trn_all_labels=[]
    vldte_all_data=[]
    vldte_all_labels=[]
    tst_all_data=[]
    tst_all_labels=[]
    #for loading train dataset, iterate through the directory to get matchig data file
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            m=r_data_file.match(f)
            if m:
                relpath = os.path.join(root, f)
                d = unpickle(relpath)
                if d:
                    trn_all_data.append(d['data'])
                    trn_all_labels.append(d['labels'])
    trn_all_data, trn_all_labels = (np.concatenate(trn_all_data).astype(np.float32),
                                          np.concatenate(trn_all_labels).astype(np.int32)
                                        )

    test_temp=unpickle(os.path.join(dataset_dir, 'test_batch'))
    vldte_all_data=test_temp['data'][0:(n_validate_samples+n_test_samples), :]
    vldte_all_labels=test_temp['labels'][0:(n_validate_samples+n_test_samples)]
    vldte_all_data, vldte_all_labels =  (np.concatenate([vldte_all_data]).astype(np.float32),
                                             np.concatenate([vldte_all_labels]).astype(np.int32))
     #transform the test images in the same manner as the train images
    trn_all_data, trn_all_labels = prepare_input(data=trn_all_data, labels=trn_all_labels)
    validate_and_test_data, validate_and_test_labels = prepare_input(data=vldte_all_data, labels=vldte_all_labels)

    vldte_all_data = validate_and_test_data[0:n_validate_samples, :, :, :]
    vldte_all_labels = validate_and_test_labels[0:n_validate_samples]
    tst_all_data = validate_and_test_data[n_validate_samples:(n_validate_samples+n_test_samples), :, :, :]
    tst_all_labels = validate_and_test_labels[n_validate_samples:(n_validate_samples+n_test_samples)]

    return (trn_all_data[start:start+length],
            trn_all_labels[start:start+length],
            vldte_all_data,
            vldte_all_labels,
            tst_all_data,
            tst_all_labels)
