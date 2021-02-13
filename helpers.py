#
#  Helper functions for Fashion MNIST Analysis
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#  This function is after  https://github.com/zalandoresearch/fashion-mnist
#
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_FashionMNIST(data_path):
    Xtrn, Ytrn = load_mnist(data_path, "train")
    Xtst, Ytst = load_mnist(data_path, "t10k")
    return Xtrn.astype(np.float), Ytrn, Xtst.astype(np.float), Ytst


def EuclideanCalculator(data_ds):
    '''
    This function receives the dataset with all features and classes
    and returns:
    1. A list with 9 dataframes each sorted by Euclidean distance of each
    observation with respect to the mean in ascending order.
    2. A Dictionary with the mean of each class.
    '''

    mean_dict = {}
    class_dfs = []
    
    for class_ in sorted(list(set(data_ds['class']))):
        distances = []
        class_df = data_ds[data_ds['class'] == class_].iloc[:,:-1]
        
        
        class_mean = class_df.values.mean(axis=0)
        mean_dict[class_] = class_mean

        
        for i in range(len(class_df)):
            distances.append(np.linalg.norm(class_df.iloc[i, :].values-class_mean))
        
        class_df.insert(0, 'euclidean_dist', distances)
        class_dfs.append(class_df.sort_values(by='euclidean_dist'))
        
    return class_dfs, mean_dict

def NormMeanCalculator(normed_data_ds):
    '''
    This function takes as input the normalized dataset of images and classes 
    and returns the mean of each class as a dictionnary.
    '''
    mean_dict = {}
    
    for class_ in sorted(list(set(normed_data_ds['class']))):
        class_df = normed_data_ds[normed_data_ds['class'] == class_].iloc[:,:-1]
        
        
        class_mean = class_df.values.mean(axis=0)
        mean_dict[class_] = class_mean
        
    return mean_dict

def MeanCalculator(data_ds):
    '''
    This function takes as input the dataset of vectorized speech and classes 
    and returns the mean of each class as a dictionnary.
    '''
    mean_dict = {}
    
    for class_ in sorted(list(set(data_ds['class']))):
        class_df = data_ds[data_ds['class'] == class_].iloc[:,:-1]
        
        
        class_mean = class_df.values.mean(axis=0)
        mean_dict[class_] = class_mean
        
    return mean_dict


def PlotCkGrid(ck_vec):
    # Plot the grid of images
    fig, axes = plt.subplots(1,3, figsize = (12,4), sharex=True, sharey = True, constrained_layout = True )
    
    # First 10 principal components
    
    #axes[i,0].set_ylabel("Class "+ str(i))
    axes[0].imshow(ck_vec[0].reshape(28,28), aspect='auto', cmap = "gray_r")
    axes[0].set_xlabel('k=1', fontsize = 15)
    axes[1].imshow(ck_vec[1].reshape(28,28), aspect='auto', cmap = "gray_r")
    axes[1].set_xlabel('k=2', fontsize = 15)
    axes[2].imshow(ck_vec[2].reshape(28,28), aspect='auto', cmap = "gray_r")
    axes[2].set_xlabel('k=3', fontsize = 15)
    
    # Define Grid title
    fig.suptitle("Ck vectors from Direct Cosine Transform", y=1.01, fontweight = 'bold', fontsize = 15)
    plt.tight_layout()
    
    return plt.show()

def NormalizeData(conf_matrix):
    TP = (conf_matrix[0][0])/(conf_matrix[0][0]+conf_matrix[1][0])
    FP = (conf_matrix[1][0])/(conf_matrix[0][0]+conf_matrix[1][0])
    FN = (conf_matrix[0][1])/(conf_matrix[0][1]+conf_matrix[1][1])
    TN = (conf_matrix[1][1])/(conf_matrix[0][1]+conf_matrix[1][1])
    
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print('Accuracy:', accuracy)

    return np.array(([TP, FN],[FP,TN]))

def PlotConfusionMatrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    plt.subplots(figsize=(10,12))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),   range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)