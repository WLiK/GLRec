# base on NCF model: https://github.com/guoyang9/NCF

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import pandas as pd
from sklearn.metrics import roc_auc_score
# from torchinfo import summary
# from tensorboardX import SummaryWriter

import model_bert
import config
import evaluate
import data_utils
from transformers import AutoTokenizer, BertModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
max_len = 256

def metrics_AUC(model, test_loader, tokenizer):
    pre_score, gold = [], []

    for user, item, label in test_loader:
        user_inputs = tokenizer(user, return_tensors="pt", padding="max_length", max_length=max_len,truncation=True)
        # user_inputs = bert_model(**user_inputs)
        # user_inputs = user_inputs.pooler_output
        item_inputs = tokenizer(item, return_tensors="pt",padding="max_length", max_length=max_len, truncation=True)
        # item_inputs = bert_model(**item_inputs)
        # item_inputs = item_inputs.pooler_output
        # # print(np.shape(user_inputs))
        # user_inputs = user_inputs.cuda()
        # item_inputs = item_inputs.cuda()
        cv_input_ids = user_inputs['input_ids'].cuda()
        # print(cv_input_ids.shape)
        cv_attention_mask = user_inputs['attention_mask'].cuda()
        cv_token_type_ids = user_inputs['token_type_ids'].cuda()
        # user_inputs = bert_model(**user_inputs)
        # user_inputs = user_inputs.pooler_output
        # item_inputs = tokenizer(item, return_tensors="pt",padding=True, truncation=True)
        jd_input_ids = item_inputs['input_ids'].cuda()
        jd_attention_mask = item_inputs['attention_mask'].cuda()
        jd_token_type_ids = item_inputs['token_type_ids'].cuda()
        label = label.cpu().numpy()

        # predictions = model(user_inputs, item_inputs)
        predictions = model(cv_input_ids, cv_attention_mask, cv_token_type_ids, jd_input_ids, jd_attention_mask, jd_token_type_ids)
        predictions = predictions.detach().cpu().numpy()
        pre_score.extend(predictions)
        gold.extend(label)
        # print(pre_score)
    print(pre_score)
    print(gold)
    print(len(gold))
    return roc_auc_score(gold, pre_score)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', default='/code/wulikang/bigmodel/code_q/zhengzhi/graph_train_large_ood.csv', type=str, help='')
parser.add_argument('--val_path', default='/code/wulikang/bigmodel/code_q/zhengzhi/graph_valid_large_ood.csv', type=str, help='')
parser.add_argument("--lr", 
    type=float, 
    default=0.001, 
    help="learning rate")
parser.add_argument("--dropout", 
    type=float,
    default=0.0,  
    help="dropout rate")
parser.add_argument("--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=200,  
    help="training epoches")
parser.add_argument("--top_k", 
    type=int, 
    default=10, 
    help="compute metrics@top_k")
parser.add_argument("--factor_num", 
    type=int,
    default=128, 
    help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
    type=int,
    default=2, 
    help="number of layers in MLP model")
parser.add_argument("--num_ng", 
    type=int,
    default=4, 
    help="sample negative items for training")
parser.add_argument("--test_num_ng", 
    type=int,
    default=99, 
    help="sample part of negative items for testing")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="0",  
    help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

bert_dir = "/code/zhengzhi/LLM_models/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(bert_dir)
print('bert token')


train_data_path = args.train_path
val_data_path = args.val_path

train_data = pd.read_csv(train_data_path)
train_dataset = data_utils.NCFData(train_data)

val_data = pd.read_csv(val_data_path)
val_dataset = data_utils.NCFData(val_data)
print(len(train_data))
print(len(val_data))
# print(train_data)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=False,num_workers=1)
test_loader=DataLoader(val_dataset,batch_size=32,shuffle=True,drop_last=False,num_workers=1)

print("-------")

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None
    

input_emb = 768
model = model_bert.CVJDTwoTowerMLP(model_path = bert_dir)
print('--------')
model.cuda()
# summary(net,input_size=(64,2,768))
print(model)
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
loss_function = nn.BCEWithLogitsLoss()

# if config.model == 'NeuMF-pre':
#     optimizer = optim.SGD(model.parameters(), lr=args.lr)
# else:
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr)ß
optimizer = optim.SGD(model.parameters(), lr=args.lr)


# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
print_count = 100
mean_loss = 0.0
for epoch in range(args.epochs):
    model.train() # Enable dropout (if have).
    start_time = time.time()
    # train_loader.dataset.ng_sample()
    # mean_loss = 0.0
    for user, item, label in train_loader:
        # print(user)
        user_inputs = tokenizer(user, return_tensors="pt",padding="max_length", max_length=max_len, truncation=True)
        # print(user_inputs)
        # user_id = user_inputs['input_ids'].cuda()
        cv_input_ids = user_inputs['input_ids'].cuda()
        # print(cv_input_ids.shape)
        cv_attention_mask = user_inputs['attention_mask'].cuda()
        cv_token_type_ids = user_inputs['token_type_ids'].cuda()
        # user_inputs = bert_model(**user_inputs)
        # user_inputs = user_inputs.pooler_output
        item_inputs = tokenizer(item, return_tensors="pt",padding="max_length", max_length=max_len, truncation=True)
        jd_input_ids = item_inputs['input_ids'].cuda()
        jd_attention_mask = item_inputs['attention_mask'].cuda()
        jd_token_type_ids = item_inputs['token_type_ids'].cuda()
        
        label = label.float().cuda()

        
        # prediction = model(user_inputs, item_inputs)
        prediction = model(cv_input_ids, cv_attention_mask, cv_token_type_ids, jd_input_ids, jd_attention_mask, jd_token_type_ids)
        loss = loss_function(prediction, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        mean_loss += loss.item()
        count += 1
        if(count % print_count == 0):
            mean_loss = mean_loss / float(print_count)
            print(mean_loss)
            mean_loss = 0.0
            

    model.eval()
    # HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
    AUC = metrics_AUC(model, test_loader, tokenizer)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("AUC: {:.3f}".format(AUC))
