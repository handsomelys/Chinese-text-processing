from numpy import *
from math import *

'''
loadDataSet(fileName)函数将文本文件导入到一个列表中，
文本文件每一行为tab分隔的浮点数，
每一个列表会被添加到dataMat中，最后返回dataMat，
该返回值是一个包含许多其他列表的列表
'''

def densify(x,n):
    d = [0.] * n
    for i , v in enumerate(x):
        d[i] = v
    return d

def norm(x):
    #返回向量的模
    return int(math.sqrt(sum(e**2 for e in x)))

def normalize(x):
    #标准化向量
    a = np.array(x)
    normalize_x = a/norm(x)
    return normalize_x


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')

        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


'''
distEclud(vecA, vecB)函数计算两个向量的欧式距离
公式：sqrt((x1-x2)^2+(y1-y2)^2)
'''


def distEclud(vecA, vecB):
    return math.sqrt(sum(power(vecA - vecB, 2)))

"""
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
"""
def cos_sim(vecA, vecB):
    vecA = np.mat(vecA)
    vector_b = np.mat(vecB)
    num = float(vecA * vecB.T)
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
#为什么 sim = 0.5 + 0.5 * cos
'''
randCent()函数为给定数据集构建一个包含k个随机质心的集合。
随机质心必须要在整个数据集的边界之内，这可以通过找到数据集每一维的最小值和最大值来完成。
然后生成0到1.0之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
'''


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # 创建存储k个质心的矩阵
    #遍历每一维
    for j in range(n):  # 在边界范围内，随机生成k个质心 
        minJ = min(dataSet[:, j])  # 边界的最小值   [:,j]   每行的第j个元素
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 边界范围   max-min
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids

#四个参数(数据集,簇的数量,计算距离和创建初始质心的函数是可选的)
def kMeans(dataSet, k, distMeas=distEclud, createRandCent=randCent):
    
	N = shape(dataSet)[0]   #N行数据
    #簇分配结果矩阵(包括两列:一列是记录簇的索引值第二列是存储误差(当前点到簇质心的距离))
	clusterAssment = mat(zeros((N,2)))  #与dataset行数一样 但是有两列
    #初始化k个质心
	centroids = createRandCent(dataSet, k)
    # 迭代标志,为true则继续迭代 即簇不再改变时停止
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(N):
			minDist = inf
			minIndex = -1
            #计算与各个质心之间的距离 选取最近的一个
			for j in range(k):
				distIJ = distMeas(centroids[j,:], dataSet[i,:]) #计算数据点到质心的距离 [j,:] 第j行样本数据
				if distIJ < minDist:    #如果距离比minDist还小 更新minDist和最小簇心索引
					minDist = distIJ
					minIndex = j
            #更新每一行样本所属的簇
			if clusterAssment[i,0] != minIndex: #簇分配结果改变
				clusterChanged = True   #簇改变
			clusterAssment[i,:] = minIndex, minDist**2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
		#print(centroids)
        #更新质心
		for cent in range(k):
            #????
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]   #获取该簇中的所有点
            #(clusterAssment[:,0].A==cent)[0]   返回属于cent簇的每个样本的坐标点
			centroids[cent,:] = mean(ptsInClust, axis=0)    #质心修改为簇中所有点的平均值，mean 就是求平均值的
            #axis=0表示输出矩阵是1行，也就是求每一列的平均值 axis=1表示输出矩阵是1列, 也就是求每一行的平均值
	return centroids, clusterAssment


'''
二分K-均值聚类算法
'''


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  # 确定数据集中数据点的总数
    # 创建一个矩阵来存放每个点的簇分配结果，包含两列：一列是记录簇索引值，第二列是存储误差。
    # 误差是指当前点到簇质心的距离，后面将使用该误差来评价聚类的效果。
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 计算整个数据集的质心，即初始时的质心的坐标为所有数据点的均值
    centList = [centroid0]  # 创建一个初始化只要一个初始质心的列表

    # 计算所有数据点到初始质心的距离平方误差
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    # 该while循环不断地对簇进行划分，直到得到设定的簇数目为止
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):  # 对每一个质心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :] # 获取当前簇 i 下的所有数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2,
                                                distMeas)  # 通过KMeans()函数，得到生成两个质心的簇，即二分，获取到质心及其每个簇的误差值
            # 将二分kMeans结果中的平方和的距离进行求和
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])# 将未参与二分kMeans分配结果中的平方和的距离进行求和
            #print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            #总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果越好
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #这两句👴没看懂
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 调用二分kMeans的结果，默认簇是0,1
        #在kmeans处做好了划分
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 更新为最佳质心
        print(bestClustAss)
        #print('最好的质心列表是: ', bestCentToSplit)
        #print('最好的簇分配结果的长度是the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 更新原来的质心list中的第i个质心为使用二分kMeans后最好的质心的第一个质心
        centList.append(bestNewCents[1, :].tolist()[0])  # 添加最佳质心的第二个质心
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # 重新分配最好簇下的数据（质心）以及误差平方和
    return mat(centList), clusterAssment


# 画图
def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=7, color='orange')
    plt.show()


def main():
    dataMat = mat(loadDataSet('E:\\temp\\program\\python\\git_2020_dachuang\\testSet2.txt'))
    # 指定获取四个质心
    # myCentroids, clustAssing= kMeans(dataMat,4)
    matr = random.rand(500,50)
    myCentroids, clustAssing = biKmeans(matr, 4)
    
    print("--------------------------------------------------")
    print("最终的质心列表：")
    print(myCentroids)
    print("--------------------------------------------------")
    print(clustAssing)
    
    #show(dataMat, 4, myCentroids, clustAssing)

    #matr = random.rand(50,4)
    #print(matr)

'''
class Spherical_kmeans:
    def __init__(self,dataSet,k,epsilon,m,distMeas=cos_sim,createCent=randCent):
        super().__init__()
        for i in dataSet:
            dataSet[i] = normalize(i)
        self.dataSet = dataSet    
        self.k = k  #划分的簇数
        self.epsilon = epsilon  #精度
        self.m = m  #迭代次数
        self.n = shape(dataSet)[0]  #dataset样本个数
        self.centroids = createCent(self.dataSet,self.k)

    def run_spherical_kmeans(self):
        pass
'''























if __name__ == '__main__':
    main()