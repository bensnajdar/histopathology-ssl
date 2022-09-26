import umap
import umap.plot
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import numpy as np
import logging

import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def create_umap_plot(data: np.array, targets: np.array = None, reducer_seed: int = 42):
    reducer = umap.UMAP(random_state=reducer_seed)
    mapper = reducer.fit(data)
    umap_plot = umap.plot.points(mapper, labels=targets, theme='fire')
    return umap_plot.figure


def create_conf_matrix_plot(data: np.array, targets: np.array, plot_labels = 'auto'):
    conf_matrix = confusion_matrix(data, targets, normalize='all')
    conf_plot = sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=plot_labels, yticklabels=plot_labels)
    plt.tight_layout()
    return conf_plot.figure

def UMAP_vis(feature_vector_arr, label_vector_arr, set_name=None):
    
    scaled_data = StandardScaler().fit_transform(feature_vector_arr)
        
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_data)

    fig = plt.figure(figsize=(10,10))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s = 3,
        c=[sns.color_palette()[int(x)] for x in label_vector_arr])
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid()
    if set_name:
        plt.title('UMAP projection of the ('+set_name+') embedding', fontsize=18)
    else:
        plt.title('UMAP projection of the embedding', fontsize=18)
    
    return fig



def TSNE_vis(feature_vector_arr, label_vector_arr, set_name=None):
    embedding = TSNE(n_components=2).fit_transform(feature_vector_arr)

    fig = plt.figure(figsize=(10,10))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s = 3,
        c=[sns.color_palette()[int(x)] for x in label_vector_arr])
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid()
    if set_name:
        plt.title('TSNE projection of the ('+set_name+') embedding', fontsize=18)
    else:
        plt.title('TSNE projection of the embedding', fontsize=18)
    
    return fig


def TSNE_vis_plotly(feature_vector_arr, label_vector_arr, dim=2 ,set_name=None):
    
    if (dim==2):
        
        embedding = TSNE(n_components=2).fit_transform(feature_vector_arr)

        print('Embedding shape:',embedding.shape)

        color = [x for x in label_vector_arr]
        emb_df = pd.DataFrame(data={'c_0':embedding[:, 0],'c_1':embedding[:, 1],'class':color})

        print(emb_df.columns)
        if set_name:
            title = 'TSNE projection of the ('+set_name+') embedding'
        else:
            title = 'TSNE projection of the embedding'
        fig = px.scatter(emb_df, x="c_0", y="c_1", color="class", title=title, hover_data=['class'],hover_name='class')
        
    
    elif (dim==3):
        
        embedding = TSNE(n_components=3).fit_transform(feature_vector_arr)

        print('Embedding shape:',embedding.shape)

        color = [x for x in label_vector_arr]
        emb_df = pd.DataFrame(data={'c_0':embedding[:, 0],'c_1':embedding[:, 1],'c_2':embedding[:, 2],'class':color})

        print(emb_df.columns)
        if set_name:
            title = 'TSNE projection of the Kather ('+set_name+') embedding'
        else:
            title = 'TSNE projection of the Kather embedding'
        fig = px.scatter_3d(emb_df, x="c_0", y="c_1",z='c_2', color="class", title=title, hover_data=['class'],hover_name='class')
    
    fig.update_layout(
    autosize=False,
    width=800,
    height=600,
    margin=dict(l=10, r=20, t=35, b=20),
    paper_bgcolor="LightSteelBlue")
    
    return fig
    



class SaveEmbedding_Hook:
    def __init__(self):
        self.feature_vector = []
        self.label_vector = []
    
    def __call__(self,module, input, output):
        for vec in output:
            self.feature_vector.append(vec.detach().cpu().numpy())
    
    def save_label(self, label):
        for lab in label:
            self.label_vector.append(lab)
    
    def clear(self):
        self.feature_vector = []
        self.label_vector = []