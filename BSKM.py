
import numpy as np
import random
from matplotlib import pyplot
from scipy.spatial.distance import cosine

np.set_printoptions(suppress=True)
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
    
    num = float(np.matmul(x, y))
    s = np.linalg.norm(x) * np.linalg.norm(y)   #np.linalg.norm 默认是求整体矩阵元素平方和再开根号 即是模
    if s == 0:
       result = 0.0
    else:
          result = num/s
        
    return (result)
    '''
    return float(np.matmul(x,y))

def cos_dot_procuct(x,y):
    #print(np.matmul(x,y))
    #return (np.matmul(x,y))
    #print('x:',x)
    #print('y:',y)
    return cosine(x,y)
def labeling(x):

    for i in range(len(x)):
        x[i].append(int(i))
    return x

def cos_distance(x,y):
    return 1-cos_dot_procuct(x,y)

class SKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)  #数据集
        self.dataSet_labeled = labeling(self.dataSet.tolist())
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        #self.dim = np.shape(dataSet)[1] #维度

    def setDataSet(self,dataSet):
        self.dataSet = dataSet

    def initCenters(self):
        #print(np.max(np.max(a,axis=0)))
        #print(np.min(np.min(a,axis=0)))
        center = []
        n = np.shape(self.dataSet)[1]
        min_value = np.min(np.min(self.dataSet,axis=0))
        max_value = np.max(np.max(self.dataSet,axis=0))
        range_value = float(max_value-min_value)
        for i in range(n):
            center.append(float(random.uniform(min_value,max_value)))
        center = np.array(center)
        return center


    def fit(self):
        self.centers = {}   #存放质心   key:index value:point of centroid
        
        for i in range(self.k):#从data中选质心 初始化
            #print(self.dataSet)
            self.centers[i] = self.initCenters()


        
        for i in range(self.m):
            n = np.shape(self.dataSet_labeled)[1] - 1
            self.clf = {}   #每个样本归属的簇 即分组情况
            self.clf_label = {}
            for i in range(self.k):
                self.clf[i] = []
                self.clf_label[i] = []
            for feature in self.dataSet_labeled:
                distances = []
                for center in self.centers:
                    #print(type(feature))
                    #print(type(self.centers[center]))
                    #print(np.shape(feature))
                    #print(feature)
                    #print(feature[:n])
                    distances.append(cos_dot_procuct(feature[:n],self.centers[center]))  #存放feature与各个质心的cos 此时即计算点乘
                #print(distances)
                #print(type(distances))
                #np.array(distances)

                #print(distances)
                classification = distances.index(np.nanmin(distances))    #取余弦最小
                self.clf[classification].append(feature[:n])
                self.clf_label[classification].append(feature[-1])

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
                if np.sum((current_centers - origin_centers)/(origin_centers+1) *100.0) > self.epislon:
                    optimized = False
            if optimized:
                print('次数： ',i)
                break

    def clusting(self,_data):
        #预测输入的样本属于哪个类
        cos_ = [cos_similarity(_data,self.centers[center]) for center in self.centers]
        index = distances.index[np.nanmax(cos_)]
        return index

class SKM_for_BSKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        #print(type(dataSet))
        n = int(np.shape(dataSet)[1])
        label_n = int(n-1)
        l = dataSet[:,-1]
        self.tmp_dataSet = normalize_dataSet(dataSet[:,:label_n])  #数据集
        self.dataSet = np.column_stack((self.tmp_dataSet,l))
        #print(self.dataSet)
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        #self.dim = np.shape(dataSet)[1] #维度

    def setDataSet(self,dataSet):
        dataSet = np.array(dataSet)
        #print(np.shape(dataSet))
        #print(dataSet)
        n = int(np.shape(dataSet)[1])
        label_n = int(n-1)
        l = dataSet[:,-1]
        self.tmp_dataSet = normalize_dataSet(dataSet[:,:label_n])  #数据集
        self.dataSet = np.column_stack((self.tmp_dataSet,l))
        #print(self.dataSet)
    def initCenters(self):
        #print(np.max(np.max(a,axis=0)))
        #print(np.min(np.min(a,axis=0)))
        center = []
        n = np.shape(self.tmp_dataSet)[1]
        min_value = np.min(np.min(self.tmp_dataSet,axis=0))
        max_value = np.max(np.max(self.tmp_dataSet,axis=0))
        range_value = float(max_value-min_value)
        for i in range(n):
            center.append(float(random.uniform(min_value,max_value)))
        center = np.array(center)
        return center


    def fit(self):
        self.centers = {}   #存放质心   key:index value:point of centroid
        
        for i in range(self.k):#从data中选质心 初始化
            #print(self.dataSet)
            self.centers[i] = self.initCenters()

        #print(self.centers)
        
        for i in range(self.m):
            n = np.shape(self.dataSet)[1] - 1
            self.clf = {}   #每个样本归属的簇 即分组情况
            self.clf_label = {}
            for i in range(self.k):
                self.clf[i] = []
                self.clf_label[i] = []
            for feature in self.dataSet:
                distances = []
                for center in self.centers:
                    distances.append(cos_dot_procuct(feature[:n],self.centers[center]))  #存放feature与各个质心的cos 此时即计算点乘
                classification = distances.index(np.nanmin(distances))    #取余弦最小
                self.clf[classification].append(feature)
                self.clf_label[classification].append(feature[-1])

            prev_centers = dict(self.centers)   #存放原本的质心

            for c in self.clf:
                self.centers[c] = np.average(np.array(self.clf[c])[:,:n],axis=0)    #求每个簇内更新之后的质心
                self.centers[c] = normalize(self.centers[c])
                #print(np.linalg.norm(self.centers[c]))
            optimized = True    #判断是否达到精度
            for center in self.centers:
                origin_centers = prev_centers[center]   #上一次的质心
                current_centers = self.centers[center]  #当前质心
                if np.sum((current_centers - origin_centers)/(origin_centers+1) *100.0) > self.epislon:
                    optimized = False
            if optimized:

                break
            
class BSKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        n = int(np.shape(dataSet)[1])
        label_n = int(n-1)
        l = dataSet[:,-1]
        self.tmp_dataSet = normalize_dataSet(dataSet[:,:label_n])  #数据集
        self.dataSet = np.column_stack((self.tmp_dataSet,l))
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        self.dim = np.shape(dataSet)[1] #维度
        #print(self.dataSet)

        self.skm = SKM_for_BSKM(self.dataSet,2,self.m)  #SKM实例
    def fit(self):
        n = np.shape(self.dataSet)[1] - 1
        self.centers_ = {}
        self.centers_[0] = np.average(self.dataSet[:,:n],axis=0)
        self.centers_[0] = normalize(self.centers_[0])  #初始化簇心 为所有样本点的中心点
        #print('self.centers[0]:',self.centers_[0])
        self.clf_ = {}
        self.clf_[0] = []
        self.clf_label_ = {}
        self.clf_label_[0] = []
        index = 0
        self.centers = {}
        self.clf = {}
        self.clf_label = {}
        for feature in self.dataSet:
            self.clf_[0].append(feature)
            self.clf_label_[0].append(feature[-1])

        self.skm.fit()
        self.centers = self.skm.centers
        self.clf = self.skm.clf
        self.clf_label = self.skm.clf_label
        #print(self.centers)
        self.index = 1
        while(len(self.centers) < self.k):
            cos_distances_ = []
            for i in range(len(self.centers)):
                sse = -np.inf
                max_index = -1
                
                for j in range(len(self.clf)):
                    cos = 0
                    for k in range(len(self.clf[j])):
                        cos += cos_distance(np.array(self.clf[j])[k][:n],self.centers[i])
                if cos > sse:
                    sse = cos
                    max_index = j   #第j个簇的sse最小 赋值到min_index
            self.skm.setDataSet(self.clf[max_index])
            self.skm.fit()
            self.centers[self.index] = self.skm.centers[0]
            self.clf[self.index] = self.skm.clf[0]
            self.clf_label[self.index] = self.skm.clf_label[0]
            self.index = self.index + 1
            self.centers[self.index] = self.skm.centers[1]
            self.clf[self.index] = self.skm.clf[1]
            self.clf_label[self.index] = self.skm.clf_label[1]

def run_skm():
    x = np.random.rand(100,2)
    skm = SKM(x,2,300,0.0001)
    skm.fit()
    print(skm.clf_label)
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
    x = np.random.rand(100,2)
    x = labeling(x.tolist())
    x = np.array(x)
    bskm = BSKM(x,2,300,0.0001)
    bskm.fit()
    print(bskm.clf_label)
    #print(skm.clf)
    #print(skm.centers)
    #print(skm.clf)
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
    

def run_skm_modified():
    x = np.random.rand(100,2)
    x = labeling(x.tolist())
    x = np.array(x)
    skm = SKM_for_BSKM(x,2,300,0.0001)
    skm.fit()
    print(skm.clf_label)
    #print(skm.clf)
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

if __name__ == '__main__':
    #run_skm()
    run_bskm()
    #run_skm()
    #run_skm_modified()