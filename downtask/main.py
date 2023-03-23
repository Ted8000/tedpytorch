import os
import logging
import argparse
import numpy as np

from model import get_model
from dataloader import parse_data

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(message)s', filename='log', filemode='a', level=logging.DEBUG)


class Trainer:
    def __init__(self, config):
        self.config = config
        num_labels = len(config.id2label)
        self.model  = get_model(config.model_load_path, num_labels)
        self.model.to(device)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")
        self.criterion_mse = nn.MSELoss()
    
    def train(self, train_loader, dev_loader):
        self.model.train()

        loss_list = []
        pred_result = []
        label_result = []

        total_step = len(train_loader)
        step_ = len(train_loader)//3
        best_acc = 0

        self.model.config.label2id=self.config.label2id
        self.model.config.id2label=self.config.id2label

        ## model training
        print("train start...")
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_parameters, lr=config.learning_rate, eps=config.adam_epsilon)

        for epoch in range(config.epochs):
            self.model.train()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['label'].to(device)

                outputs = self.model(input_ids, attention_mask)

                loss = self.criterion_ce(outputs[0], labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                optimizer.step()

                loss_list.append(loss.cpu().item())

                if (i+1) % step_ == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, config.epochs, i+1, total_step, np.mean(loss_list)))
                
            with torch.no_grad():
                self.model.eval()
                correct = 0
                total = 0
                for i, batch in enumerate(dev_loader):
                    input_ids = batch['input_ids'].squeeze(1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(device)
                    labels = batch['label'].to(device)
                    outputs = self.model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                now_acc = 100 * correct / total
                if now_acc > best_acc:
                    self.save(self.model, config.model_save_path)
                    best_acc = now_acc
                print('Accuracy of the network on the dev data: {} %'.format(100 * correct / total))
        logging.info('Model best accuracy is {} %'.format(best_acc))
    
    def test(self, test_loader):
        best_acc = 0
        with torch.no_grad():
            self.model.eval()
            correct = 0
            total = 0
            for i, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['label'].to(device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs[0].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            now_acc = 100 * correct / total
            if now_acc > best_acc:
                self.save(self.model, './data/teacher_model.pt')
                best_acc = now_acc
            print('Accuracy of the network on the dev data: {} %'.format(100 * correct / total))
        logging.info('Model test accuracy is {} %'.format(best_acc))    

    def predict(self, data_loader):
        """
        predict
        """
        pred_result = []
        label_result = []
        best_acc = 0
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs[0].data, 1)

        logging.info('Model predict finished')   
    
    def load(self, path):
        return torch.load(path)
    
    def save(self, model, path):
        return torch.save(model, path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-token", "--tokenizer", nargs='?', const=1, type=str, default="xlm-roberta-base",
                        help="model path load")
    parser.add_argument("-m", "--model_load_path", nargs='?', const=1, type=str, default="xlm-roberta-base",
                            help="model path load")
    parser.add_argument("-s", "--model_save_path", nargs='?', const=1, type=str, default='data/model.pt',
                            help="model path save")
    parser.add_argument("-len", "--length", nargs='?', const=1, type=int, default=32,
                            help="train sentence length")
    parser.add_argument("-ep", "--epochs", nargs='?', const=1, type=int, default=10,
                            help="train epoch num")
    parser.add_argument("-bs", "--batch_size", nargs='?', const=1, type=int, default=32,
                            help="train batch size num")
    parser.add_argument("-g", "--gpu", nargs='?', const=1, type=int, default=0,
                            help="gpu device")
    parser.add_argument("-lr", "--learning_rate", nargs='?', const=1, type=int, default=3e-5,
                            help="learn rate")
    parser.add_argument("-eps","--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--log_path", default='./log_all', type=str)
    parser.add_argument("--config_path", default=None, type=str)
    parser.add_argument("--df_file", default=None, type=str)
    parser.add_argument("--task", default="cls", type=str)
    parser.add_argument("--temperature", default=4, type=float)
    parser.add_argument("--alpha", default=1, type=float)
    parser.add_argument('--seed', default=42, type=int) 
    parser.add_argument('--path', default=None, type=str)

    args = parser.parse_args()
    config = args
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu')
    
    train_loader, dev_loader, test_loader, id2label, label2id = parse_data(config)

    config.id2label = id2label
    config.label2id = label2id

    trainer = Trainer(config)
    trainer.train(train_loader, dev_loader)
    
    logging.info('Finished')
    