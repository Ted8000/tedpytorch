import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from transformers import *
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df_train = pd.read_csv('./train.txt', sep='\t', header=None)
df_test = pd.read_csv('./test.txt', sep='\t', header=None)
df_train = shuffle(df_train)
df_test = shuffle(df_test)
train_x, train_y = df_train[1].tolist(), df_train[0].tolist()
test_x, test_y = df_test[1].tolist(), df_test[0].tolist()
le = LabelEncoder()
le.fit(train_y)
train_y = le.transform(train_y)
test_y = le.transform(test_y)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    report = classification_report(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

class model_test(BertPreTrainedModel):
  def __init__(self, config):
      super().__init__(config)
      self.num_labels = config.num_labels

      self.bert = BertModel(config)
      self.dropout = nn.Dropout(config.hidden_dropout_prob)
      self.classifier = nn.Linear(config.hidden_size, config.num_labels)

      self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
  ):


      outputs = self.bert(
          input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids,
          position_ids=position_ids,
          head_mask=head_mask,
          inputs_embeds=inputs_embeds,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
      )

      pooled_output = outputs[1]

      pooled_output = self.dropout(pooled_output)
      logits = self.classifier(pooled_output)

      outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

      if labels is not None:
          if self.num_labels == 1:
              #  We are doing regression
              loss_fct = MSELoss()
              loss = loss_fct(logits.view(-1), labels.view(-1))
          else:
              loss_fct = CrossEntropyLoss()
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
          outputs = (loss,) + outputs

      return outputs  # (loss), logits, (hidden_states), (attentions)

def dummy_data_collector(features):
  batch = {}
  batch['input_ids'] = torch.stack([f[0] for f in features])
  batch['attention_mask'] = torch.stack([f[1] for f in features])
  batch['labels'] = torch.stack([f[2] for f in features])
  
  return batch


# param
num_class = 13

pretrain_model = 'voidful/albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(pretrain_model, cache_dir='../cache')
model = model_test.from_pretrained(pretrain_model, num_labels=num_class, cache_dir='../cache')


batch = tokenizer(train_x, padding='max_length', truncation=True, max_length=30, return_tensors="pt")
y_ = torch.tensor(train_y)
dataset_ = TensorDataset(batch['input_ids'], batch['attention_mask'], y_)

batch_test = tokenizer(test_x, padding='max_length', truncation=True, max_length=30, return_tensors="pt")
y_2 = torch.tensor(test_y)
dataset_test = TensorDataset(batch_test['input_ids'], batch_test['attention_mask'], y_2)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,             # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    save_steps=3000,
    evaluate_during_training=True
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset_,         # training dataset
    eval_dataset=dataset_test,
    data_collator=dummy_data_collector,         # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()

print(trainer.evaluate(dataset_test))
