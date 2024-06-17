import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from src.nn import EmbeddingDataset, SimpleNN, train, evaluate


if __name__ == '__main__':
    df = pd.read_pickle('../data/embeddings/embeddings.pkl')

    # create one-hot encoding for the labels
    df = pd.get_dummies(df, columns=['category'], dtype=int)

    # Get a list of the label columns and add a new column for the category
    label_columns = df.columns[4:]
    df['category'] = np.argmax(df[label_columns].values, axis=1)

    embeddings = np.array(df['embeddings'].values.tolist())
    labels = np.array(df["category"].values.tolist())
    
    dataset = EmbeddingDataset(embeddings, labels)
    
    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True)
    
    input_dim = embeddings.shape[1]
    hidden_dim = [512, 256]
    output_dim = np.max(labels) + 1
    
    model = SimpleNN(input_dim, hidden_dim, output_dim)
 
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
    # Train the model
    train(model, train_dataloader, criterion, optimizer, num_epochs=40)

    print("GT:", " ".join([str(x) for x in labels[:10]]))
    print("Pred:", " ".join([str(x) for x in torch.argmax(model(torch.tensor(embeddings[:10], dtype=torch.float32)), dim=1)]))
    # Evaluate the model on the test set
    acc = evaluate(model, test_dataloader, criterion)
    print(acc)
    