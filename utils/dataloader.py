
class myDataloader():
    def __init__(self, trainset, batch_size):
        self.trainset = trainset
        self.batch_size = batch_size
        self.term_num = len(trainset)
    
    def __getitem__(self, index):
        start_index = index*self.batch_size
        end_index = (index+1)*self.batch_size if (index+1)*self.batch_size < self.term_num else self.term_num
        return self.trainset[start_index:end_index] 