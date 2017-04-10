import cPickle
import numpy as np

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
