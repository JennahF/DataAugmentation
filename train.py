from utils import dataloader
from model.model import LFModel
import pickle

Epoch = 3
alpha = 0.2
batch_size = 64
class_count = 5

def loss_func(true, pred):
    return (true-pred)*(true-pred)

def third_constraint(model):
    A = connection_method()
    error = 0.
    for j in range(A.shape[0]):
        for k in range(A.shape[1]):
            if A[j][k] != 0:
                error += model.error(j,k)*alpha
    return error

def trainModel(traindata, validata, model, optimizer):

    start_time = time.time()

    def trainEpoch(epoch, traindata, model):
        batch_num = traindata.batch_num
        total_rmse = 0.
        aver_rmse = 0.
        for batch_number in range(batch_num):
            error = 0.
            batchdata = traindata[i]
            batch_size = 0
            for term in batchdata:
                i = term["userId"]
                j = term["movidId"]
                true_rank = term["rank"]
                pred_rank = model(i, j)
                error += loss_func(true_rank, pred_rank)
                batch_size += 1

            total_rmse += error
            aver_rmse += error/batch_size

            error += model.omega() + third_constraint(model)

            model.optimize(error)

            end_time = time.time()

            if batch_number % log_interval == 0:
                print( "Epoch %d, %d th batch, avg rmse: %.2f, total rmse: %6.2f; %6.0f s elapsed"
                    % (epoch, batch_number, aver_rmse, total_rmse, end_time - start_time))

    
    for i in range(Epoch):
        train_rmse = trainEpoch(traindata, model)

        valid_rmse = eval(validata, model)
        print('Epoch %d: RMSE on (train, validation): (%.2f, %.2f)' % (train_rmse, valid_rmse))


def main():
    with open('./data/trainset.pickle', 'rb') as f:
        trainset = pickle.load(f)
    with open('./data/validset.pickle', 'rb') as f:
        validset = pickle.load(f)

    batched_train_set = dataloader(trainset, batch_size)

    model = LFModel(class_count)

    trainModel(batched_train_set, validset, model, optimizer)


if __name__ == "__main__":
    main()