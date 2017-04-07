import numpy as np 
import cPickle as pickle
import tensorflow as tf
import TensorGraph as tg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
batch_size = 200
def training(data, label, valid_dataset, valid_labels):
    #gradients = tensorgraph.get_gradients(data, label)
    for epoch in range(10):
        for i in range(len(data)/batch_size):
            x = i * batch_size
            tensorgraph.optimize(data[x:x+batch_size], label[x:x+batch_size])
        print("Valid accuracy: %.1f%%" % accuracy(tensorgraph.predict(valid_dataset), valid_labels))
    return tensorgraph.get_parameters()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def open_dataset(start, length, pickle_file='/home/rche/data/notMNIST.pickle'):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
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

def clustering(X, true_k=10, minibatch = False, showLable = False):
    kmeans = KMeans(n_clusters=true_k, random_state=0).fit(X)
    return -kmeans.score(X) 

def best_cluster(X):
    true_ks = []
    scores = []
    for i in range(50):            
        score = train(X,true_k=(i+1))
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(true_ks,scores,label="error",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def draw_grad(X):
    x_val = np.arange(len(X))
    plt.figure(figsize=(8,4))
    plt.plot(x_val, X,label="value",color="red",linewidth=1)
    plt.ylabel("value")
    plt.legend()
    plt.show()

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = open_dataset(0,10000)
tensorgraph = tg.TensorGraph()

gradients = training(train_dataset,train_labels, valid_dataset, valid_labels)
#for i in xrange(len(gradients)):
#    gradients[i] = gradients[i][0]
#    print gradients[i].shape

arr_grad = np.reshape(gradients[0], 78400)
arr_grad = np.append(arr_grad, gradients[1])
arr_grad = np.append(arr_grad, np.reshape(gradients[2], 1000))
arr_grad = np.append(arr_grad, gradients[3])


#print len(arr_grad)
#arr_grad.sort()
#draw_grad(arr_grad)
