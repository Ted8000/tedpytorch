import torch
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader, Dataset, dataset, DataLoader
import pandas as pd
from sklearn.utils import shuffle
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', filename='log', filemode='a', level=logging.INFO)

class CustomDataset(Dataset):
    def __init__(self, df, label2id, length, tok):
        self.df = df
        self.length = length
        self.tokenizer = AutoTokenizer.from_pretrained(tok)
        self.train_x = self.df.iloc[:,0].tolist()
        self.train_y = list(map(lambda x:label2id[x], self.df.iloc[:,1].tolist()))
        

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=self.length, return_tensors="pt")
        batch['label'] = torch.tensor(self.train_y[index])
        return batch

def parse_data(config):
    df = pd.read_csv(config.df_file, sep='\t')
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
    
    train_set = CustomDataset(df_train, label2id, config.length, config.tokenizer)
    dev_set = CustomDataset(df_dev, label2id, config.length, config.tokenizer)
    test_set = CustomDataset(df_test, label2id, config.length, config.tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    return train_loader, dev_loader, test_loader, id2label, label2id

# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(list(map(lambda x:eval(x), examples["tokens"])), truncation=True, is_split_into_words=True)
    
#     labels = []
#     for i, label in enumerate(list(map(lambda x:eval(x), examples["ner_tags"]))):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:
#             # Special tokens have a word id that is None. We set the label to -100 so they are automatically
#             # ignored in the loss function.
#             if word_idx is None:
#                 label_ids.append(-100)
#             # We set the label for the first token of each word.
#             elif word_idx != previous_word_idx:
#                 label_ids.append(label[word_idx])
#             # For the other tokens in a word, we set the label to either the current label or -100, depending on
#             # the label_all_tokens flag.
#             else:
#                 label_ids.append(label[word_idx] if label_all_tokens else -100)
#             previous_word_idx = word_idx

#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

# def parse_ner(config):
#     data = Dataset.from_csv(config.df_file)
#     train_tokenized = data.map(tokenize_and_align_labels, batched=True)

#     d_tmp = train_tokenized.train_test_split(0.2)
#     train = d_tmp['train']
#     test = d_tmp['test']

#     return train, test