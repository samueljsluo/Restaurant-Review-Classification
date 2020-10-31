import pandas as pd
from transformers import BertTokenizer, AutoModel, AdamW
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from operator import itemgetter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU state:', device)
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 32
EPOCH = 20

bert = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        #output
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step, batch in enumerate(train_dataloader):
        if step%50==0 and not step==0:
            print("Batch {:>5} of {:>5}.".format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        model.zero_grad()
        preds = model(sent_id,mask)
        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds=preds.detach().cpu().numpy()

        total_preds.append(preds)
    avg_loss = total_loss/len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def evaluate():
    print("\nEvaluating...")
    model.eval()

    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step, batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


dataset = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset['Review'] = dataset['Review'].apply(lambda x: x.lower())

train_text, temp_text, train_labels, temp_labels = train_test_split(dataset['Review'], dataset['Liked'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=dataset['Liked'])
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

token_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=25, pad_to_max_length=True, truncation=True)
token_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=25, pad_to_max_length=True, truncation=True)
token_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length=25, pad_to_max_length=True, truncation=True)

train_seq = torch.tensor(token_train['input_ids'])
train_mask = torch.tensor(token_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

test_seq = torch.tensor(token_test['input_ids'])
test_mask = torch.tensor(token_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

val_seq = torch.tensor(token_val['input_ids'])
val_mask = torch.tensor(token_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

for param in bert.parameters():
    param.requires_grad = False

model = BERT(bert)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
cross_entropy = nn.CrossEntropyLoss()

best_valid_loss = float('inf')
train_losses = []
valid_losses = []

for epoch in range(EPOCH):
    print("\nEpoch {:} / {:}".format(epoch+1, EPOCH))
    train_loss, _ = train()
    val_loss, _ = evaluate()

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'model.pt')
    train_losses.append(train_loss)
    valid_losses.append(val_loss)
    print(f"\nTraining Loss: {train_loss:.3f}")
    print(f"Validation Loss: {best_valid_loss:.3f}")

path = 'model.pt'
model.load_state_dict(torch.load(path))

with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(classification_report(test_y, preds))