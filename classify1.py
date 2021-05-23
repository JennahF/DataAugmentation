# -*- coding: utf-8 -*-

from posix import listdir
from re import sub
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
from model.TextCNN import TextCnn
from model.BiLSTM import TextRNN
from utils.dataloader import TextDataset
import argparse
import time
import logging
import pickle
from sklearn import metrics
from collections import Counter

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use_aug', type=int, default=0)
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--modelnum', type=int, default='0')
parser.add_argument('--l2', type=float, default=0.001)
parser.add_argument('--modelname', type=str, default='TextCNN')
parser.add_argument('--resultnum', type=int, default=0)
parser.add_argument('--modelfilename', type=str, default="")
args = parser.parse_args()

resultnum = args.resultnum

# 创建logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建handler
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = 'logs/'
if args.use_aug:
    logfile = log_path+rq+"_"+str(resultnum)+"_"+args.modelname+"_"+args.dataset+"_3.log"
else:
    logfile = log_path+rq+"_"+str(resultnum)+"_"+args.modelname+"_"+args.dataset+"_2.log"
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.DEBUG) 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
#定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(console)

torch.manual_seed(args.seed)

# Create the configuration
config = Config(cuda=args.gpu,
                epoch2=args.epoch,
                learning_rate2=args.lr,
                batch_size=args.batch_size,
                dataset = args.dataset)
                
logging.info(args)

train_embeddings = './temp/word2vec/'+args.dataset+"_model2_original.pickle"

def read_test_data():
    test_embeddings = './temp/word2vec/'+args.dataset+'_test.pickle'
    with open(test_embeddings, 'rb') as f:
        testset0 = pickle.load(f) #n*1*sentence_length*embedding_dim

    testset = testset0[0]
    label = testset0[1]
    print("test:",Counter(label))
    print("test set len:", len(testset))
    label = torch.tensor(label)
    lens = torch.tensor([len(i[0]) for i in testset])
    testset_tensor = torch.zeros(len(testset), max(lens), len(testset[0][0][0]))
    for i, mat in enumerate(testset):
        for j in range(lens[i]):
            testset_tensor[i][j] = torch.Tensor(testset[i][0][j])
    testset_tensor = testset_tensor.unsqueeze(1)

    new_index = torch.randperm(len(testset_tensor))
    testset_tensor = testset_tensor[new_index]
    lens = lens[new_index]
    label = label[new_index]

    l = testset_tensor.size()[0]
    if args.dataset == 'IMDB':
        subset = int(0.4*l)
    else:
        subset = l

    return testset_tensor[:subset].cuda(), lens[:subset].cuda(), label[:subset].cuda()

def evaluate(model, testset, lens, label):
    with torch.no_grad():
        # print(testset.size(), lens.size(), label.size())
        out = model(testset, lens)
        preds = F.softmax(out, dim=1).argmax(1)
        # print(label, label.size())
        # auc = metrics.roc_auc_score(label.cpu().numpy(),preds.cpu().numpy())
        auc = torch.tensor(0)
        acc = (torch.eq(preds, label)).float().mean()
        return acc, auc

total_loss_list = []
total_acc_list = []
train_acc_list = []

def train(testset, test_lens, test_label, subset):
    training_iter = TextDataset(train_embeddings, args.batch_size, True, subset)

    if args.use_aug:
        checkpoint = torch.load('model/'+ args.modelname +'/'+args.dataset+"/"+args.modelfilename)
        logger.info(checkpoint['para'].__dict__)
        if args.modelname == 'TextCNN':
            model = TextCnn(checkpoint['para'], 'gpu')
        elif args.modelname == 'TextRNN':
            model = TextRNN(checkpoint['para'])

        model.load_state_dict(checkpoint['model'])
    else:
        if args.modelname == 'TextCNN':
            model = TextCnn(config, 'gpu')
        elif args.modelname == 'TextRNN':
            model = TextRNN(config)
        logger.info(config.__dict__)

    if torch.cuda.is_available():
        model = model.cuda()

    test_score, test_auc = evaluate(model, testset, test_lens, test_label)
    logging.info("before training: {}".format(test_score))
    total_acc_list.append(str(round(test_score.item(), 3)))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate2, weight_decay=args.l2)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate2)

    count = 0
    loss_sum = 0
    train_score = 0
    train_auc = 0
    startime = time.time()
    totaltime = 0
    # Train the model
    logger.info("...start training...")
    for epoch in range(config.Epoch2):
        for index in  range(training_iter.batch_num):
            data, lens, label = training_iter[index]
            if config.cuda and torch.cuda.is_available():
                data = data.cuda()
                lens = lens.cuda()
                label = label.cuda()
            
            out = model(data, lens)
            loss = criterion(out, label)

            loss_sum += loss.item()
            count += 1

            # if count % 10 == 0:
            #     test_score = evaluate(model, testset, lens, test_label)
            #     logger.info("epoch %d The loss is: %.5f, test accuracy: %.4f" %(epoch, (loss_sum/10), test_score))
            #     loss_sum = 0
            #     count = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, auc = evaluate(model, data, lens, label)
            train_score += acc.item()
            train_auc += auc.item()
        
        test_score, test_auc = evaluate(model, testset, test_lens, test_label)
        logger.info( "Epoch %d, toal_loss %.4f, aver loss %.4f, train accuracy %.4f, train auc %.4f, test accuracy %.4f, test auc %.4f"
                % (epoch, loss_sum, loss_sum/count, train_score/training_iter.batch_num, train_auc/training_iter.batch_num, test_score, test_auc))
        total_loss_list.append(str(round(loss_sum,3)))
        total_acc_list.append(str(round(test_score.item(),3)))
        train_acc_list.append(str(round(train_score/training_iter.batch_num, 3)))
        loss_sum = 0
        train_score = 0
        train_auc = 0
    
    if args.use_aug:
        torch.save(model.state_dict(), 'model/'+ args.modelname +'/'+ str(args.resultnum) + '_epoch{}_aug_{}.ckpt'.format(epoch, subset))
    else:
        torch.save(model.state_dict(), 'model/'+ args.modelname +'/'+ str(args.resultnum) + '_epoch{}.ckpt_{}'.format(epoch, subset))

testset, test_lens, test_label = read_test_data()
print(testset.size(), test_lens.size(), test_label.size())

subset_list = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# subset_list = [1]
for subset in subset_list:
    print("\nusing subset: ", subset)
    train(testset, test_lens, test_label, subset)

if args.use_aug:
    total_loss = ','.join(total_loss_list)
    total_acc = ','.join(total_acc_list)
    train_acc = ','.join(train_acc_list)
    with open("results/r"+str(resultnum)+".txt", 'a') as f:
        f.write(total_loss+"\n"+total_acc+"\n"+train_acc+'\n')
else:
    total_loss = ','.join(total_loss_list)
    total_acc = ','.join(total_acc_list)
    train_acc = ','.join(train_acc_list)
    with open("results/r"+str(resultnum)+".txt", 'a') as f:
        f.write(total_loss+"\n"+total_acc+"\n"+train_acc+'\n')

if args.use_aug:
    with open("results/r"+str(resultnum)+".txt", 'a') as f:
        f.write(args.modelname+","+args.dataset)

