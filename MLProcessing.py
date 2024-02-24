import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap


def read_file(path):
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, encoding='cp1251')
    return df


def normalization(data):
    std_data = ((data - data.mean())/data.std())
    return std_data/np.max(abs(std_data))


def cleaning(data_):
    data = data_
    column_name = data.columns
    i = 0
    for column in column_name:
        if i == 0:
            df_filtered = data.loc[(data[column] >= (data[column].mean() - 3 * data[column].std())) &
                               (data[column] <= (data[column].mean() + 3 * data[column].std()))]
            i+=1
        else:
            df_filtered = df_filtered.loc[(df_filtered[column] >= (df_filtered[column].mean() - 3 * df_filtered[column].std()))
                                        & (df_filtered[column] <= (df_filtered[column].mean() + 3 * df_filtered[column].std()))]
    df_filtered = df_filtered.reset_index(drop = 'True')
    return df_filtered


def data_preparation(data):
    data_not_nan = data.dropna().reset_index(drop=True)
    data_ = cleaning(data_not_nan)
    column_name = data_.columns
    for column in column_name:
        if (column != 'X') & (column != 'Y'):
            data_[column] = normalization(data_[column])
    return data_


def gistograms(data):
    size = len(data.columns)
    column_name = data.columns
    fig, axs = plt.subplots(size - 2, figsize=(10, 70))
    i = 0
    for column in column_name:
        if (column != 'X') & (column !='Y'):
            axs[i].hist(data[column], bins=30)
            axs[i].set_title(column)
            i+=1
    plt.show()
    

def cards(data, marker_size=0.5, limit=1):
    X = data['X']
    Y = data['Y']
    column_name = data.columns
    fig, axs = plt.subplots(len(data.columns)-2, figsize=(10, 80))
    i = 0
    for column in column_name:
        if (column != 'X') & (column != 'Y'):
            im = axs[i].scatter(X, Y, c = data[column], s = marker_size, cmap = 'turbo', vmin = data[column].min()/limit, vmax = data[column].max()/limit)
            axs[i].set_title(column)
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            plt.colorbar(im, ax=axs[i])
            i+=1
    plt.show()

    
def cor_matrix(data):
    f, ax = plt.subplots(dpi = 150)
    corr = data.corr()
    sns.heatmap(np.abs(corr),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            vmin=0.0, vmax=1.0,
            square=True, ax=ax)
    plt.show()

    
def del_cors(data, cor_size):
    data_del = data.copy()
    corr_matrix = data_del.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > cor_size)] 
    data_del.drop(to_drop, axis = 1, inplace = True)
    return data_del


def prepare_classters(data):
    data_ = data.drop(columns = ['X', 'Y'], axis = 1)
    return data_


def clasters(data, klasster_size):
    kmeans = KMeans(n_clusters = klasster_size)
    kmeans.fit(data)
    return kmeans 



def pca(data, size_components):
    pca = PCA(n_components = size_components)
    pca_data = pca.fit_transform(data)
    return pca_data


def classter_labels(data, classter):
    fig = plt.figure(figsize=(10, 7))
    X = data['X']
    Y = data['Y']
    plt.scatter(X, Y, c = classter.labels_, s = 0.5, cmap = 'turbo')
    plt.title('класстер({classter_size})'.format(classter_size = np.bincount(classter.labels_).size))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()
    

def pca_classter_labels(data, pca_data, classter):
    fig = plt.figure(figsize=(10, 7))
    X = data['X']
    Y = data['Y']
    num_rows, num_cols = pca_data.shape
    plt.scatter(X, Y, c = classter.labels_, s = 0.5, cmap = 'turbo')
    plt.title('pca({pca_size}) classter({classter_size})'.format(pca_size = num_cols, classter_size = np.bincount(classter.labels_).size))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()
    
    
def umap_classter_labels(data, umap_data, classter):
    fig = plt.figure(figsize=(10, 7))
    X = data['X']
    Y = data['Y']
    num_rows, num_cols = umap_data.shape
    plt.scatter(X, Y, c = classter.labels_, s = 0.5, cmap = 'turbo')
    plt.title('umap({pca_size}) classter({classter_size})'.format(pca_size = num_cols, classter_size = np.bincount(classter.labels_).size))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()


def umap_classter(data, size):
    reducer = umap.UMAP(size)
    embedding = reducer.fit_transform(data)
    return embedding


def chip(data):
    fig = plt.figure(dpi = 150)
    plt.scatter(data[:, 0], data[:, 1], s = 0.05, cmap = 'turbo')
    plt.colorbar()

