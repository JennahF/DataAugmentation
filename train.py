from typing import Text

from numpy.lib.function_base import average
from utils.dataloader import myDataloader
import pickle
from config import Config
from model.TextCNN import TextCnn
from model.BiLSTM import TextRNN
from tqdm import tqdm
import torch
import time
import argparse
import logging
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--runmode', type=str, default='train')
parser.add_argument('--modelnum', type=str, default='0')
parser.add_argument('--l2', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=8)
parser.add_argument('--dataset', type=str, default='IMDB')
parser.add_argument('--negNum', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--maxbound', type=float, default=1)
parser.add_argument('--modelname', type=str, default='TextCNN')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--posNum', type=int, default=1)
parser.add_argument('--resultnum', type=int, default=0)
parser.add_argument('--modelfilename', type=str, default="")
parser.add_argument('--loadmodel', type=int, default=0)
parser.add_argument('--startepoch', type=int, default=0)
args = parser.parse_args()

resultnum = args.resultnum

# 创建logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建handler
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = 'logs/'
logfile = log_path+rq+"_"+str(resultnum)+"_"+args.modelname+"_"+args.dataset+"_1.log"
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

para = Config(dataset=args.dataset,
              negSampleNum=args.negNum,
              learning_rate1=args.lr,
              batch_size=args.bs,
              max_bound=args.maxbound,
              epoch1=args.epoch,
              posSampleNum=args.posNum)

logger.info(para.__dict__)
logger.info(args)

def read_test_data():
    test_embeddings = './temp/word2vec/'+args.dataset+'_test.pickle'
    with open(test_embeddings, 'rb') as f:
        testset0 = pickle.load(f) #n*1*sentence_length*embedding_dim

    testset = testset0[0]
    label = testset0[1]
    # print("test:",Counter(label))
    # print("test set len:", len(testset))
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

def contrastive_loss(output, neg_num, pos_Num):
    loss = 0.
    count0 = 0.
    count1 = 0
    neg_loss = 0.
    pos_loss = 0.
    for i in range(0, len(output), 1+pos_Num+neg_num):
        base_sample = output[i]
        pos_sample = output[i+1:i+1+pos_Num]
        neg_sample = output[i+2+pos_Num:i+2+pos_Num+neg_num]
        for pos in pos_sample:
            count0+=1
            pos_loss+=torch.norm(base_sample-pos)
            loss += torch.norm(base_sample-pos)*torch.norm(base_sample-pos)
        for j,neg in enumerate(neg_sample):
            count1 += 1
            neg_loss += torch.norm(base_sample-neg)
            loss += max(0, para.max_bound-torch.norm(base_sample-neg))*max(0, para.max_bound-torch.norm(base_sample-neg))

    return loss, neg_loss/count1, pos_loss/count0

def trainModel(traindata, model, optimizer):

    start_time = time.time()
    loss_list = []
    test_acc_list = []
    test_acc_list_float = []

    testset, test_lens, test_label = read_test_data()

    def trainEpoch(epoch, traindata, model):
        total_loss = 0.
        neg_loss = 0.
        pos_loss = 0.
        for batch_number in tqdm(range(traindata.batch_number)):
            batch_data, batch_lens = traindata[batch_number]
            if torch.cuda.is_available():
                batch_data = batch_data.cuda()
                batch_lens = batch_lens.cuda()

            output = model(batch_data, batch_lens)

            loss, aver_neg_loss, aver_pos_loss = contrastive_loss(F.softmax(output, dim=1), para.negSampleNum, para.poSampleNum)
            neg_loss += aver_neg_loss.item()
            pos_loss += aver_pos_loss.item()
            # logger.info(loss.item(), end=" ")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = time.time()

            total_loss+=loss.item()

        test_score, _ = evaluate(model, testset, test_lens, test_label)
        logger.info( "Epoch %d, %d th batch, total loss: %.2f, aver loss: %6.2f, aver neg loss: %6.2f, aver pos loss: %6.2f, test acc: %6.2f; %6.0f s elapsed"
                % (epoch, batch_number, total_loss, total_loss/(batch_number+1), neg_loss/(batch_number+1), pos_loss/(batch_number+1), test_score, end_time - start_time))
        loss_list.append(str(round(total_loss,3)))
        test_acc_list.append(str(round(test_score.item(), 3)))
        test_acc_list_float.append(round(test_score.item(), 3))
        
    
    for i in range(para.Epoch1):
        trainEpoch(i+args.startepoch, traindata, model)

        checkpoint = {
            'model': model.state_dict(),
            'para': para
        }
        # torch.save(checkpoint, './model/' + args.modelname +'/'+args.dataset+'/'+str(args.resultnum)+'_Epoch_'+str(i+args.startepoch)+'_model_'+str(args.posNum)+'_'+str(args.negNum)+'_'+str(round(average(test_acc_list_float), 3))+'.pt')
        torch.save(checkpoint, './model/' + args.modelname +'/'+args.dataset+'/'+str(args.resultnum)+'_Epoch_'+str(i+args.startepoch)+'_model_'+str(args.posNum)+'_'+str(args.negNum)+'.pt')

    logger.info("model saved!")

    loss_list = ','.join(loss_list)
    test_acc_list = ','.join(test_acc_list)
    # with open("results/r"+str(resultnum)+".txt", 'a') as f:
    #     f.write(str(args.posNum)+","+str(args.negNum)+","+str(max(test_acc_list_float))+","+str(min(test_acc_list_float))+","+str(average(test_acc_list_float))+'\n')
    with open("results/r"+str(resultnum)+".txt", 'a') as f:
        f.write(loss_list+'\n'+test_acc_list+'\n')


    x0 = [i+1 for i in range(len(test_acc_list_float))]
    plt.plot(x0, test_acc_list_float)
    plt.ylim(0,1)
    plt.savefig('./model/' + args.modelname +'/'+args.dataset+'/img/'+str(args.resultnum)+'.png')

def generate_dataset(model):
    with torch.no_grad():
        aug_embed_save_files = './temp/word2vec/'+args.dataset+'_model2_augmented.pickle'
        embeddings_save_files = './temp/word2vec/'+args.dataset+"_model2_original.pickle"

        with open(embeddings_save_files, 'rb') as f:
            trainset0 = pickle.load(f) 
        label0 = trainset0[1]
        trainset0 = trainset0[0]

        lens = torch.tensor([len(i[0]) for i in trainset0])
        # logger.info(type(max(lens)), type(len(testset[0][0][0])))
        trainset = torch.zeros(len(trainset0), max(lens), len(trainset0[0][0][0]))
        for index, mat in enumerate(trainset):
            for j in range(lens[index]):
                trainset[index][j] = torch.Tensor(trainset0[index][0][j])
        logger.info(trainset.size(), lens.size())
        augdata = model(trainset, lens)
        logger.info(trainset.size(),augdata.size())
        trainset = trainset.to('cpu')
        augdata = augdata.to('cpu')

        #n*1*sentence_length*embedding_dim
        out = [torch.cat((trainset.unsqueeze(1), augdata.unsqueeze(1)), 0), torch.cat((lens, lens), 0), torch.tensor(label0+label0)]

        with open(aug_embed_save_files, 'wb') as v:
            pickle.dump(out, v)

def main(posNum=""):

    if args.runmode == 'train':
        
        if posNum != "":
            train_embeddings = './temp/word2vec/'+args.dataset+"_model1_"+posNum+".pickle"
        else:
            train_embeddings = './temp/word2vec/'+args.dataset+"_model1"+".pickle"
        with open(train_embeddings, 'rb') as f:
            data = pickle.load(f)
        logger.info("train data loaded!")

        batched_train_set = myDataloader(data[0], data[1], para.batch_size, para.negSampleNum, para.poSampleNum)


        if args.loadmodel:
            checkpoint = torch.load('model/'+ args.modelname +'/'+args.dataset+"/"+args.modelfilename)
            if args.modelname == 'TextCNN':
                model = TextCnn(checkpoint['para'], 'gpu')
            elif args.modelname == 'TextRNN':
                model = TextRNN(checkpoint['para'])
            model.load_state_dict(checkpoint['model'])
        else:
            if args.modelname == 'TextCNN':
                model = TextCnn(para, 'gpu')
            elif args.modelname == 'TextRNN':
                model = TextRNN(para)

        if torch.cuda.is_available():
            model = model.cuda()

        # optimizer = torch.optim.Adam(model.parameters(), lr=para.learning_rate1, weight_decay=args.l2)
        optimizer = torch.optim.Adam(model.parameters(), lr=para.learning_rate1)

        
        logger.info("...start training...")
        trainModel(batched_train_set, model, optimizer)

    elif args.runmode == 'generate':
        checkpoint = torch.load("./model/Seq2Seq/model_"+args.modelnum+".pt")
        model = Seq2Seq(checkpoint['para'], 'gpu')
        model.load_state_dict(checkpoint['model'])

        generate_dataset(model)


if __name__ == "__main__":
    # pos_neg_list = [[1,7],[2,6],[3,5],[4,4],[5,3],[6,2],[7,1]]
    # for pos_neg in pos_neg_list:
    #     pos = pos_neg[0]
    #     neg = pos_neg[1]
    #     logger.info("pos,neg Num={},{}".format(pos, neg))
    #     args.posNum = pos
    #     args.negNum = neg
    #     main(str(pos))
    main(str(args.posNum))