class parameters():
    batch_size = 64
    embedding_dim = 300
    ENC_EMB_DIM = 300
    HID_DIM = 256
    N_LAYERS = 1
    ENC_DROPOUT = 0.1
    OUTPUT_DIM = 300
    DEC_EMB_DIM = 256
    DEC_DROPOUT = 0.1
    negSampleNum = 2
    Epoch = 10
    max_bound = 150
    learning_rate = 0.001

class Config(object):
    def __init__(self, word_embedding_dimension=300, epoch1=5, epoch2=10, 
                cuda=True,learning_rate1=0.00001, learning_rate2=0.00001, 
                batch_size=64, negSampleNum = 3, max_bound=1,dataset='IMDB', 
                hidden_size = 256, hidden_layers = 1, bidirectional = True,posSampleNum=1):
        self.word_embedding_dimension = word_embedding_dimension     # 词向量的维度
        self.Epoch1 = epoch1                                          # 遍历样本次数
        self.Epoch2 = epoch2
        self.dropout = 0.1
        
        if dataset in ['IMDB', 'Subj', 'PC', 'CoLA']:
            self.label_num = 2
        elif dataset == 'TREC':
            self.label_num = 6

        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.batch_size = batch_size
        self.cuda = cuda

        #pre-trained model paras
        self.negSampleNum = negSampleNum
        self.poSampleNum = posSampleNum
        self.max_bound = max_bound

        #TextCNN paras
        self.kernel_sizes = [3,4,5]
        
        #Bi-LSTM paras
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.bidirectional = bidirectional

