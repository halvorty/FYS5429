from transformers import AutoTokenizer, AutoModel
model_id = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
 
 
import torch
import numpy as np
import pandas as pd

# Import needed for function pooling
from typing import Dict
 
# Function from mixedbread-ai/mxbai-embed-large-v1 model card
def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()



# Load the data
df = pd.read_csv('../data/news_uk_dataset.csv')


# Split the data into 1001 subsets and embed each subset
import pickle as pkl
length = len(df)
subset_size = length // 100
rest = length - subset_size * 100

# Embed the first 99 subsets
for i in range(0,99):
    subset = df.iloc[i*subset_size:(i+1)*subset_size]
    with torch.no_grad():
        inputs = tokenizer(subset['title'].tolist(), padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = pooling(outputs.last_hidden_state, inputs)
    subset.insert(len(df.columns), 'embeddings', embeddings.tolist())
    with open(f'../embedded_datafiles/embeddings_{i}.pkl', 'wb') as f:
        pkl.dump(subset, f)
subset = df.iloc[100*subset_size:]

# Embed the last subset
with torch.no_grad():
    inputs = tokenizer(subset['title'].tolist(), padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = pooling(outputs.last_hidden_state, inputs)
subset.insert(len(df.columns), 'embeddings', embeddings.tolist())
with open(f'../embedded_datafiles/embeddings_{100}.pkl', 'wb') as f:
    pkl.dump(subset, f)

# Combine the subsets 
df = pd.read_pickle('../embeddings/embeddings_0.pkl')
for i in range(1,100 + 1):
    df = pd.concat([df, pd.read_pickle('../embeddings/embeddings_{}.pkl'.format(i))])

# Save the combined dataframe
df.to_pickle('../embeddings/embedded_data.pkl')

# Now delete the subsets
import os
for i in range(0,100 + 1):
    os.remove('../embeddings/embeddings_{}.pkl'.format(i))
