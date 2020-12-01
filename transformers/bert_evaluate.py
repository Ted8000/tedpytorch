import torch
import numpy as np
from transformers import *
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader, Dataset, dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import copy
import sys
import os
from tqdm import tqdm
import argparse

# run command
# python bert_evaluate.py -d data/test.csv -s data/tmp.csv -m models/roberta.pt -t /data0/zhouyue/ted/data/cache/roberta/


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--path1", nargs='?', const=1, type=str, default="data/test",
                        help="test path load")
parser.add_argument("-s", "--path2", nargs='?', const=1, type=str, default="data/result.csv",
                        help="data path save")
parser.add_argument("-g", "--gpu", nargs='?', const=1, type=int, default=0,
                        help="model path save")
parser.add_argument("-bs", "--bach_size", nargs='?', const=1, type=int, default=256,
                        help="epoch num")
parser.add_argument("-t", "--tokenizer", nargs='?', const=1, type=str, default="/data0/zhouyue/ted/data/cache/roberta/",
                        help="tokenizer path load")
parser.add_argument("-m", "--pre_train_model", nargs='?', const=1, type=str, default="/data0/zhouyue/ted/data/cache/roberta/",
                        help="pre_trained model path load")
parser.add_argument("-l", "--label_path", nargs='?', const=1, type=str, default="label_save",
                        help="label path load")
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

# sentence process
df_test = pd.read_csv(args.path1, sep='\t', header=None)
test_x, test_y = df_test.iloc[:,0].tolist(), df_test.iloc[:,1].tolist()

# label load
if os.path.exists('label_save'):
    with open('label_save', 'r') as f:
        labels = f.read().split()
else:
    labels = set(train_y)|set(test_y)

label2id = {}
id2label = {}
for k,v in enumerate(labels):
    id2label[k] = v
    label2id[v] = k
    
test_y = list(map(lambda x:label2id[x], test_y))

# model load
class BERTClass(torch.nn.Module):
    def __init__(self, pretrain_name, num_class, cache_dir=None):
        super(BERTClass, self).__init__()
        if cache_dir == None:
            self.l1 = BertModel.from_pretrained(pretrain_name)
        else:
            self.l1 = BertModel.from_pretrained(pretrain_name, cache_dir=cache_dir)
        hidden = self.l1.pooler.dense.out_features
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden, num_class)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
class CustomDataset(Dataset):
    def __init__(self, tokenizer, train_x, train_y):
        self.tokenizer = tokenizer
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        batch['label'] = torch.tensor(self.train_y[index])
        return batch

num_class = len(set(labels))
batch_size = 256

tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
model = torch.load(args.pre_train_model)

test_set = CustomDataset(tokenizer, test_x, test_y)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

model.to(device)


def torch_predict(model, test_x, test_y):
    model.eval()
    
    y_pred = []
    y_real = []
    with torch.no_grad():
        correct=0
        total=0
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, token_type_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu())
            y_real.extend(labels.cpu())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    a = []
    b = []
    for i in range(len(y_pred)):
        a.append(y_pred[i].tolist())
        b.append(y_real[i].tolist())
    a = list(map(lambda x:id2label[x], a))
    b = list(map(lambda x:id2label[x], b))
    
    print(classification_report(b, a))
    print('Accuracy of the network on the valid data: {} %'.format(100 * correct / total))
    return b, a, test_x, test_y
    
y_real, y_pred, test_x, test_y = torch_predict(model, test_x, test_y)
df = pd.DataFrame()
df['msg']=test_x
df['y_real']=y_real
df['y_pred']=y_pred
df['test_y']=test_y
df.to_csv(args.path2, index=False, sep='\t')
