
import numpy as np
import random
from matplotlib import pyplot
from scipy.spatial.distance import cosine
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
    return cosine(x,y)
def labeling(x):

    for i in range(len(x)):
        x[i].append(int(i))
    return np.array(x)

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

class BSKM:
    def __init__(self,dataSet,k,m,epislon=0.0001):
        super().__init__()
        self.dataSet = normalize_dataSet(dataSet)  #数据集

        self.dataSet_labeled = labeling(self.dataSet.tolist())  #[-1]是标签
        
        #self.dataSet = dataSet
        self.k = k  #簇的个数
        self.epislon = epislon  #精度
        self.m = m  #迭代次数
        self.size = np.shape(dataSet)[0]    #粒子个数
        #self.dim = np.shape(dataSet)[1] - 1#维度
        
        #print(self.dataSet)

        self.skm = SKM(self.dataSet,2,self.m)  #SKM实例
    def fit(self):
        n = np.shape(self.dataSet_labeled)[1] - 1
        self.centers_ = {}
        self.centers_[0] = np.average(self.dataSet,axis=0)
        self.centers_[0] = normalize(self.centers_[0])  #初始化簇心 为所有样本点的中心点
        self.clf_ = {}
        self.clf_[0] = []
        self.clf_label_ = {}
        self.clf_label_[0] = []
        index = 0
        self.centers = {}
        self.clf = {}
        self.clf_label = {}
        for feature in self.dataSet_labeled:
            self.clf_[0].append(feature[:n])
            self.clf_label_[0].append(feature[-1])
        #print(len(self.clf_[0]))
        while(len(self.centers) < self.k):
            #print(len(self.centers))
            #print(self.k)
            #print('len of centers:',len(self.centers))
            #重复 直至出现4个簇
            cos_distances_ = []
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
            choosed_class_ = cos_distances_.index(np.nanmax(cos_distances_))    #获得SCE最大的簇
            #print(choosed_class_)
            
            self.skm.setDataSet(self.clf_[choosed_class_])   
            #print(len(self.clf_[choosed_class_]))
            #print((self.clf_[choosed_class_]))
            self.skm.fit()
                #note : 应该用一个临时变量存储skm分出的SCE较大簇的聚类结果
            temp_centers_ = self.skm.centers
            temp_clf_ = self.skm.clf
            temp_clf_label_ = self.skm.clf_label
            cos_distances = []
            for clf in temp_clf_: #遍历每个clf
                cos_ = 0
                for j in range(len(temp_clf_[clf])):  #遍历每个clf中的每个样本
                    for center in temp_centers_:    #计算SCE    #此处float转换出现nan 需要解决
                        #print('temp_clf_[clf][j]:',temp_clf_[clf][j])
                        #print('temp_centers_[center]:',temp_centers_[center])
                        cos_ += cos_distance(temp_clf_[clf][j],temp_centers_[center])
                        #print('cos_',cos_)
                cos_distances.append((cos_))
            #print('cos_distances:',cos_distances)
            choosed_class = cos_distances.index(np.nanmin(cos_distances)) #选取SCE最小的簇
            self.centers[index] = temp_centers_[choosed_class]
            self.clf[index] = temp_clf_[choosed_class]
            self.clf_ = self.clf
            self.clf_label[index] = temp_clf_label_[choosed_class]
            self.clf_label_ = self.clf_label
            self.centers_ = self.centers
            index += 1    



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
    
    #print(type(x))
    #print(type(y))
    #print(x.shape)
    #print(y.shape)
    bskm = BSKM(x,2,300,0.0001)
    bskm.fit()
    print(bskm.clf_label)
    #print((bskm.centers))
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
    #run_bskm()
    run_skm()