import torch
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader, Dataset, dataset, DataLoader
import pandas as pd
from sklearn.utils import shuffle
import time
import pickle
import logging

class CustomDataset(Dataset):
    def __init__(self, df, label2id, length=32):
        self.df = df
        self.length = length
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.train_x = self.df.iloc[:,0].tolist()
        self.train_y = list(map(lambda x:label2id[x], self.df.iloc[:,1].tolist()))
        

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=self.length, return_tensors="pt")
        batch['label'] = torch.tensor(self.train_y[index])
        return batch

def parse_data(path, batch_size, length):
    df = pd.read_csv(path, sep='\t')
    label_names = df.columns.values.tolist()[1:]
    
    num=int(len(df)*0.8)
    df_train = df[:num]
    df_dev = df[num:]
    df_test = df[num:]
    
    # label load
    le = LabelEncoder()
    le.fit(df.iloc[:,1].tolist())
    
    id2label={}
    label2id={}
    for i,x in enumerate(le.classes_.tolist()):
        id2label[i]=x
        label2id[x]=i
    
    train_set = CustomDataset(df_train, label2id, length)
    dev_set = CustomDataset(df_dev, label2id, length)
    test_set = CustomDataset(df_test, label2id, length)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, dev_loader, test_loader, id2label, label2id
