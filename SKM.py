import numpy as np
import random
def normalize(x):
    #标准化向量
    #print(np.linalg.norm(a[0]))
    norm = np.linalg.norm(x)
    y = x/norm
    return y

def normalize_dataSet(dataset):
    for i in len(dataset):
        dataset[i] = normalize(dataset[i])
    return dataset
        

def cos_similarity(x,y):
    #计算余弦相似度
    num = float(np.matmul(x, y))
    s = np.linalg.norm(x) * np.linalg.norm(y)   #np.linalg.norm 默认是求整体矩阵元素平方和再开根号
    if s == 0:
       result = 0.0
    else:
          result = num/s
        
    return float(result)

def mat_adder(x, y):
    #矩阵相加
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return [[m+n for m,n in zip(i,j)] for i, j in zip(x,y)]

class SKM:
    def __init__(self,dataSet,k,epislon,m):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)  #数据集
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        self.dim = np.shape(dataSet)[1] #维度

    def skm(self):
        #随机取质心 从样本中取k个
        self.centroids = random.sample(self.dataSet,self.k)
        self.cluster_assment = [[] for row in range(self.k)] #存放每个簇里面存了啥玩意儿
        for i in self.dataSet:
            max_cos = -np.inf
            max_index = -1
            for j in self.centroids:
                value = cos_similarity(i,j)
                if value > max_cos:
                    max_cos = value
                    max_index = j   #代表i∈j
            cluster_assment[max_index].append(i)
        
        #更新簇心
        for i in range(len(self.cluster_assment)):   #range(len(cluster_assment)) == self.k
            centroid_ = np.sum(self.cluster_assment[i])
            centroid_ = centroid / len(self.cluster_assment[i])
            centroid_new = normalize(centroid_) #标准化簇心
            self.centroids[i] = centroid_new
        #return self.centroids,self.cluster_assment
    def clusting(self):
        for i in range(self.m):
            self.skm()

class BSPM:
    def __init__(self):
        super().__init__()
        




