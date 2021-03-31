import os
from tqdm import tqdm
import matplotlib.pyplot as plt 


def main():
    dirs=['./data/aclImdb_v1/aclImdb/train/pos/',
        './data/aclImdb_v1/aclImdb/train/neg/',
        './data/aclImdb_v1/aclImdb/test/pos/',
        './data/aclImdb_v1/aclImdb/test/neg/']


    files = [[dir + file for file in os.listdir(dir)] for dir in dirs]

    maxlen = 2470
    statistic = [0] * 25
    for i in range(len(files)):
        for file in tqdm(files[i]):
            with open(file, 'r', encoding='utf8') as f:
                review = f.readlines()
                words = review[0].lower().split()
                statistic[int(len(words)/100)] += 1
    for i in range(len(statistic)-1):
        statistic[i+1] += statistic[i]
    X = [i for i in range(25)]
    fig = plt.figure()
    plt.bar(X,statistic,0.4,color="green")
    
    plt.show()

# for dir in dirs:
    # for _, _, files in os.walk(dir):

if __name__ == '__main__':
    main()