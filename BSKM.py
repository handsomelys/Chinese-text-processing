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
    s = np.linalg.norm(x) * np.linalg.norm(y)   #np.linalg.norm 默认是求整体矩阵元素平方和再开根号 即是模
    if s == 0:
       result = 0.0
    else:
          result = num/s
        
    return float(result)

def cos_dot_procuct(x,y):
    return float(np.matmul(x,y))

def mat_adder(x, y):
    #矩阵相加
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return [[m+n for m,n in zip(i,j)] for i, j in zip(x,y)]

def cos_distance(x,y):
    return 1-cos_dot_procuct(x,y)

class SKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)  #数据集
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        self.dim = np.shape(dataSet)[1] #维度

    def setDataSet(self,dataSet):
        self.dataSet = dataSet

    def fit(self):
        self.centers = {}   #存放质心   key:index value:point of centroid
        for i in range(self.k):#从data中选质心 初始化
            #print(self.centers)
            #print(len(self.dataSet))
            #print(self.dataSet)
            self.centers[i] = self.dataSet[i]

        for i in range(self.m):
            self.clf = {}   #每个样本归属的簇 即分组情况
            for i in range(self.k):
                self.clf[i] = []
            for feature in self.dataSet:
                distances = []
                for center in self.centers:
                    distances.append(cos_dot_procuct(feature,self.centers[center]))  #存放feature与各个质心的cos 此时即计算点乘
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
        #预测输入的样本属于哪个类
        cos_ = [cos_similarity(_data,self.centers[center]) for center in self.centers]
        index = distances.index[max(cos_)]
        return index

class BSKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)  #数据集
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        self.dim = np.shape(dataSet)[1] #维度
        

        self.skm = SKM(self.dataSet,2,self.m)  #SKM实例
    def fit(self):
        self.centers_ = {}
        self.centers_[0] = np.average(self.dataSet,axis=0)
        self.centers_[0] = normalize(self.centers_[0])  #初始化簇心 为所有样本点的中心点
        self.clf_ = {}
        self.clf_[0] = []
        index = 0
        self.centers = {}
        self.clf = {}
        for feature in self.dataSet:
            self.clf_[0].append(feature)
        #print(len(self.clf_[0]))
        while(len(self.centers) < self.k):
            #print(len(self.centers))
            #print(self.k)
            #print('len of centers:',len(self.centers))
            #重复 直至出现4个簇
            cos_distances_ = []
            '''
            for clf in self.clf_:
                cos = 0
                for j in range(len(self.clf_[clf])):
                    for center in self.centers_:
                        #print(self.clf_[clf][j])
                        cos += cos_distance(self.clf_[clf][j],self.centers_[center])
            #理论上 cosdistances里放的是k个sce和        
                print(cos)    
                cos_distances_.append(cos)
            '''
            for center in self.centers_:
                #print(len(self.centers_))
                cos = 0
                for clf in self.clf_:
                    for j in range(len(self.clf_[clf])):
                        cos += cos_distance(self.clf_[clf][j],self.centers_[center])
                        #print(cos)
                        #print(cos_distances_)
                cos_distances_.append(cos)
                #print(cos)
            #print(cos_distances_)
            choosed_class_ = cos_distances_.index(max(cos_distances_))    #获得SCE最大的簇
            #print(choosed_class_)
            
            self.skm.setDataSet(self.clf_[choosed_class_])   
            self.skm.fit()
                #note : 应该用一个临时变量存储skm分出的SCE较大簇的聚类结果
            temp_centers_ = self.skm.centers
            temp_clf_ = self.skm.clf
            cos_distances = []
            for clf in temp_clf_: #遍历每个clf
                cos_ = 0
                for j in range(len(temp_clf_[clf])):  #遍历每个clf中的每个样本
                    for center in temp_centers_:    #计算SCE
                        cos_ += cos_distance(temp_clf_[clf][j],temp_centers_[center])
                cos_distances.append(cos_)
            choosed_class = cos_distances.index(min(cos_distances)) #选取SCE最小的簇
            self.centers[index] = temp_centers_[choosed_class]
            self.clf[index] = temp_clf_[choosed_class]
            self.clf_ = self.clf
            self.centers_ = self.centers
            index += 1    



def run_skm():
    x = np.random.rand(500,2)
    skm = SKM(x,2,300,0.0001)
    skm.fit()
    
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

def run_bskm():
    x = np.random.rand(500,2)
    bskm = BSKM(x,2,300,0.0001)
    bskm.fit()
    print((bskm.centers))
    #print(skm.clf)
    #print(bskm.centers)
    #for i in bskm.clf:
        #print(bskm.clf[i])
    
    flag = 1
    for center in bskm.centers:
        if flag == 1:
            pyplot.scatter(bskm.centers[center][0],bskm.centers[center][1],marker='*',s=200,c='r')
            flag = 0
        elif flag == 0:
            pyplot.scatter(bskm.centers[center][0],bskm.centers[center][1],marker='*',s=200,c='b')
    
    for catter in bskm.clf:
        for point in bskm.clf[catter]:
            if catter == 0:
                pyplot.scatter(point[0],point[1],c='r')
            else:
                pyplot.scatter(point[0],point[1],c='b')
    pyplot.show()
    
if __name__ == '__main__':
    #run_skm()
    run_bskm()
    run_skm()