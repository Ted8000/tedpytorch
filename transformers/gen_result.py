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
from tqdm import tqdm
import argparse

# run command
# python gen_result.py -d data/raw.txt -s data/tmp.csv -m models/roberta.pt -t /data0/zhouyue/ted/data/cache/roberta/


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
test_x= df_test.iloc[:,0].tolist()

# label load
with open('label_save', 'r') as f:
    labels = f.read().split()

label2id = {}
id2label = {}
for k,v in enumerate(labels):
    id2label[k] = v
    label2id[v] = k
    

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
    def __init__(self, tokenizer, train_x):
        self.tokenizer = tokenizer
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        return batch

num_class = len(set(labels))
batch_size = 256

tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
model = torch.load(args.pre_train_model)

test_set = CustomDataset(tokenizer, test_x)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

model.to(device)


def torch_predict(model, test_x):
    model.eval()
    
    pred_conf = []
    pred_three = []
    sfmx = nn.Softmax(1)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids, token_type_ids, attention_mask)
            conf_three, predicted_three = torch.topk(sfmx(outputs.data), 3, 1)
            pred_conf.extend(conf_three.cpu())
            pred_three.extend(predicted_three.cpu())

    three_conf = []
    one_conf = []
    three_pred = []
    one_pred = []
    for i in range(len(pred_conf)):
        three_conf.append(pred_conf[i].tolist())
        one_conf.append(pred_conf[i].tolist()[0])
        three_pred.append(pred_three[i].tolist())
        one_pred.append(pred_three[i].tolist()[0])
        
    one_pred = list(map(lambda x:id2label[x], one_pred))
    three_pred = list(map(lambda x:list(map(lambda t:id2label[t], x)), three_pred))
    
    return one_pred, one_conf, three_pred, three_conf, test_x
    
one_pred, one_conf, three_pred, three_conf, test_x = torch_predict(model, test_x)
df = pd.DataFrame()
df['msg']=test_x
df['one_pred']=one_pred
df['one_conf']=one_conf
df['three_pred']=three_pred
df['three_conf']=three_conf
df.to_csv(args.path2, index=False, sep='\t')
