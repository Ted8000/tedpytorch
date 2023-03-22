import logging
import argparse
import os

from model import get_model
from dataloader import parse_data

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(message)s', filename='log', filemode='a', level=logging.DEBUG)


class Trainer:
    def __init__(self):
        num_labels = len(args.id2label)
        self.student = get_model(args.teacher_path, num_labels)
        self.teacher = get_model(args.student_path, num_labels)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")
        self.criterion_mse = nn.MSELoss()
    
    def train():
        pass
    
    def train_one_phrase(self, data_loader, dev_loader):
        self.student.train()
        
#         loss_list = []
        pred_result = []
        label_result = []
        
        total_step = len(data_loader)
        step_ = len(data_loader)//3
        best_acc = 0
        
        ## teacher stage
        print("teacher stage start...")
        self.teacher.config.label2id=label2id
        self.teacher.config.id2label=id2label
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_teacher_parameters = [
            {'params': [p for n, p in self.teacher.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.teacher.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_teacher = AdamW(optimizer_teacher_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        for epoch in range(args.epochs):
            self.teacher.train()
            for i, batch in enumerate(data_loader):
                optimizer_teacher.zero_grad()

                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['label'].to(device)

                outputs = self.teacher(input_ids, attention_mask)

                loss = self.criterion_ce(outputs[0], labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), args.max_grad_norm)
                optimizer_teacher.step()

    #             loss_list.append(loss.cpu().item())

                if (i+1) % step_ == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, args.epochs, i+1, total_step, loss.item()))
                
            with torch.no_grad():
                self.teacher.eval()
                correct = 0
                total = 0
                for i, batch in enumerate(dev_loader):
                    input_ids = batch['input_ids'].squeeze(1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(device)
                    labels = batch['label'].to(device)
                    outputs = self.teacher(input_ids, attention_mask)
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                now_acc = 100 * correct / total
                if now_acc > best_acc:
                    self.save(self.teacher, './data/teacher_model.pt')
                    best_acc = now_acc
                print('Accuracy of the network on the dev data: {} %'.format(100 * correct / total))
        logging.info('teacher stage best accuracy is {} %'.format(best_acc))
        self.teacher = self.load('./data/teacher_model.pt')
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        ## student stage
        print("student stage start...")
        self.student.config.label2id=label2id
        self.student.config.id2label=id2label
        optimizer_student_parameters = [
            {'params': [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_student = AdamW(optimizer_student_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        T = args.temperature
        best_acc = 0

        for epoch in range(args.epochs):
            self.student.train()
            for i, batch in enumerate(data_loader):
                optimizer_student.zero_grad()
                # Move tensors to the configured device
                input_ids = batch['input_ids'].squeeze(1).to(device)
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                output_s = self.teacher(input_ids, attention_mask)
                output_t = self.student(input_ids, attention_mask)
                
                # hard label loss
                loss1 = self.criterion_ce(output_s[0], labels)
                
                # soft label loss
                alpha = args.alpha
                output_S = F.log_softmax(output_s[0]/T, dim=1)
                output_T = F.softmax(output_t[0]/T, dim=1)
                loss2 = self.criterion_kl(output_S, output_T)
        #         loss2 = self.criterion_mse(output_s[0], output_t[0])
            
                # sum loss
                loss = loss1*(1-alpha) + loss2*alpha
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), args.max_grad_norm)
                optimizer_student.step()
                
                if (i+1) % step_ == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, args.epochs, i+1, total_step, loss.item()))
            
            with torch.no_grad():
                # dev
                self.student.eval()
                correct = 0
                total = 0
                for i, batch in enumerate(dev_loader):
                    input_ids = batch['input_ids'].squeeze(1).to(device)
                    attention_mask = batch['attention_mask'].squeeze(1).to(device)
                    labels = batch['label'].to(device)
                    outputs = self.student(input_ids, attention_mask)
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                now_acc = 100 * correct / total
                if now_acc > best_acc:
                    self.save(self.teacher, './data/student_model.pt')
                    best_acc = now_acc
                print('Accuracy of the network on the dev data: {} %'.format(100 * correct / total))
        logging.info('student stage best accuracy is {} %'.format(best_acc))

    def test():
        pass
    
    def predict():
        pass
    
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
    parser.add_argument("-s", "--model_save_path", nargs='?', const=1, type=str, default=None,
                            help="model path save")
    parser.add_argument("-len", "--length", nargs='?', const=1, type=int, default=32,
                            help="train sentence length")
    parser.add_argument("-ep", "--epochs", nargs='?', const=1, type=int, default=10,
                            help="train epoch num")
    parser.add_argument("-bs", "--batch_size", nargs='?', const=1, type=int, default=32,
                            help="train batch size num")
    parser.add_argument("-g", "--gpu", nargs='?', const=1, type=int, default=0,
                            help="model path save")
    parser.add_argument("-lr", "--learning_rate", nargs='?', const=1, type=int, default=3e-5,
                            help="model path save")
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
    parser.add_argument('--teacher_path', default=None, type=str)
    parser.add_argument('--student_path', default=None, type=str)

    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_loader, dev_loader, test_loader, id2label, label2id = parse_data(args.df_file, args.batch_size, args.length)

    args.id2label = id2label
    args.label2id = label2id

    trainer = Trainer()
    trainer.train_one_phrase(train_loader, dev_loader)
    
    logging.info('Accuracy')
    