import numpy as np
import random
from matplotlib import pyplot
def normalize(x):
    #标准化向量
    #print(np.linalg.norm(a[0]))
    norm = np.linalg.norm(x)
    y = x/norm
    return y

def normalize_dataSet(dataset):
    for i in range(len(dataset)):
        dataset[i] = normalize(dataset[i])
    return dataset
        

def cos_similarity(x,y):
    #计算余弦相似度
    '''
    print('x: ')
    print(x.shape)
    print('y: ')
    print(y.shape)
    '''
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
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        self.dim = np.shape(dataSet)[1] #维度

    def fit(self):
        self.centers = {}   #存放质心   key:index value:point of centroid
        for i in range(self.k):#从data中选质心 初始化
            self.centers[i] = self.dataSet[i]

        for i in range(self.m):
            self.clf = {}   #每个样本归属的簇 即分组情况
            for i in range(self.k):
                self.clf[i] = []
            for feature in self.dataSet:
                distances = []
                for center in self.centers:
                    distances.append(cos_similarity(feature,self.centers[center]))
                #print(distances)
                classification = distances.index(min(distances))    #取余弦最小
                self.clf[classification].append(feature)

            prev_centers = dict(self.centers)   #存放原本的质心
            #print(self.centers)
            #print('------------------')
            #print(prev_centers)
            for c in self.clf:
                self.centers[c] = np.average(self.clf[c],axis=0)    #求每个簇内更新之后的质心
                self.centers[c] = normalize(self.centers[c])
                #print(np.linalg.norm(self.centers[c]))
            optimized = True    #判断是否达到精度
            for center in self.centers:
                origin_centers = prev_centers[center]   #上一次的质心
                current_centers = self.centers[center]  #当前质心
                if np.sum((current_centers - origin_centers)/origin_centers *100.0) > self.epislon:
                    optimized = False
            if optimized:
                print('次数： ',i)
                break

    def clusting(self,_data):
        cos_ = [cos_similarity(_data,self.centers[center]) for center in self.centers]
        index = distances.index[max(cos_)]
        return index

class BSPM:
    def __init__(self,dataSet,k,epislon,m):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)
        self.k = k
        self.epislon = epislon
        self.m = m

        

if __name__ == '__main__':
    #x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
    x = np.random.rand(500,2)
    skm = SKM(x,2,0.0001,300)
    skm.fit()
    #print(skm.centers)
    #print(skm.clf)
    flag = 1
    for center in skm.centers:
        if flag == 1:
            pyplot.scatter(skm.centers[center][0],skm.centers[center][1],marker='*',s=200,c='r')
            flag = 0
        elif flag == 0:
            pyplot.scatter(skm.centers[center][0],skm.centers[center][1],marker='*',s=200,c='b')
    
    for catter in skm.clf:
        for point in skm.clf[catter]:
            if catter == 0:
                pyplot.scatter(point[0],point[1],c='r')
            else:
                pyplot.scatter(point[0],point[1],c='b')
            

    pyplot.show()




