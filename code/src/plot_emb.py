from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns

def reduce_emb(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    red = tsne.fit_transform(embeddings)
    x = red[:,0]
    y = red[:,1]
    return x, y

def plot_reduced_embeddibgs(x, y, labels):
    fig, ax = plt.figure()
    sns.set_theme()
    sns.scatterplot(x=x, y=y, ax=ax)
    # Add x, and y labels
    colors = sns.color_palette("tab10")
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    
    
    