import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

# Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.embeddings)
 
    def __getitem__(self, idx):
        sample = self.embeddings[idx]
        label = self.labels[idx]
        return sample, label

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        
        # Choose the activation function as relu
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        
        self.fchidden = [nn.Linear(hidden_dim[i], hidden_dim[i+1]) for i in range(len(hidden_dim) - 1)]
        self.fcout = nn.Linear(hidden_dim[-1], output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.fchidden:
            x = self.relu(layer(x))
        x = self.fcout(x)
        return x

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print("Finished Training")

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    df = pd.read_pickle('../../data/embeddings/embeddings.pkl')

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
    train(model, train_dataloader, criterion, optimizer, num_epochs=100)

    # Evaluate the model on the test set
    acc = evaluate(model, test_dataloader, criterion)
    print(acc)