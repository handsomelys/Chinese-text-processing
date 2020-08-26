import math
import random
import numpy as np
from sklearn import datasets
from scipy.spatial.distance import cosine

np.set_printoptions(suppress=True)

def cosine_similarity(x,y):
    return cosine(x,y)

def labeling(x):

    for i in range(len(x)):
        x[i].append(int(i))
    return x

def normalize(input_data):
    for i in range(len(input_data)):
        mod = np.linalg.norm(input_data[i])
        input_data[i] = input_data[i] / mod
    # print(input_data)
    return input_data

def get_closest_dist(data,centroids):
    min_dist = np.inf
    for i,centroid in enumerate(centroids):
        #print(data)
        #print(centroid)
        dist = cosine_similarity(data,centroid)
        #print(dist)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def Skmeanspp_centroid(data,k):
    m,n = np.shape(data)
    cluster_centers = []
    cluster_centers.append(random.choice(data[:,:-1]))
    category = []
    d = [0 for _ in range(m)]
    for _ in range(1,k):
        total = 0.0
        for i,point in enumerate(data):
            #print(point)
            #print(point[:-1])
            d[i] = get_closest_dist(point[:-1],cluster_centers)
            total += d[i]
        total *= random.random()
        for i,di in enumerate(d):
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data[i,:-1])
            break
    return cluster_centers

def Skmeanspp(data, k, omiga, max_iter):
    '''根据SKMeans算法求解聚类中心
        input:  data(mat):训练数据
                k(int):类别个数
                omiga(float):精度要求
                max_iter(int):最大迭代次数
        output: centroids(mat):训练完成的聚类中心
                subCenter(mat):每一个样本所属的类别
    '''
    #subCenter 第0维放的是簇的属类 (0 or 1)
    m, n = np.shape(data)  # m：样本的个数，n：特征的维度
    subCenter = np.mat(np.zeros((m, n + 1)))  # 初始化每一个样本所属的类别
    centroids = np.mat(normalize(Skmeanspp_centroid(data, k)))  # 随机初始化聚类中心
    data[:, :-1] = normalize(data[:, :-1])  # 标准化数据
    for iter in range(max_iter):
        # centroids = normalize(randCent(data, k))  # 随机初始化聚类中心
        for i in range(m):  # 更新隶属度
            max_index = 0
            max_value = -np.inf
            for j in range(k):  # 遍历全部聚类中心，求出最大值即属于哪个聚类中心
                if (data[i, :-1] * centroids[j].T)[0, 0] > max_value:   #cosine
                    max_value = (data[i, :-1] * centroids[j].T)[0, 0]  # 求样本和聚类中心的内积，得到最大值
                    max_index = j  # 得到最大值的下标即聚类中心的编号
            lst = [max_index]  # 将样本聚类标签和样本一同写入subCenter矩阵中
            #print(type(lst))
            #print(type(data[i].tolist()))
            #print(data[i].tolist())
            lst.extend(data[i].tolist())
            
            #print('data[i].tolist()',data[i].tolist())
            #print((data[i][0]==data[i]).any())     true 实际上这两个是一样的
            #print('data[i].tolist()[0]',data[i].tolist()[0])
            subCenter[i,] = np.mat(lst)
            #print(subCenter)
        # 更新簇中心向量
        update_center_vector = [[] for _ in range(k)]  # 用于存储每个簇的向量计算结果
        update_center_count = [0 for _ in range(k)]  # 用于存储每个簇的样本个数
        for i in range(m):
            center_index = int(subCenter[i, 0])  # 获取簇中心的编号
            update_center_count[center_index] += 1  # 对应簇的样本个数+1
            try:  # 将对应簇的样本数据累加
                update_center_vector[center_index] = update_center_vector[center_index] + subCenter[i, 1:-1]
            except:
                update_center_vector[center_index] = subCenter[i, 1:-1]
        centroids_next = np.mat(np.zeros((k, n - 1)))  # 方法参照randCent函数
        try:
            for i in range(k):  # 更新每一簇的聚类中心坐标
                m_vector = np.mat(update_center_vector[i]) / float(update_center_count[i])
                centroids_next[i, :] = normalize(m_vector)
        except:
            continue
        # print('centroids_next',centroids_next)
        # print("centroids",centroids)
        # print("="*30)
        # 矩阵运算求出 max|c(j,t+1)-c(j,t)|是否 <omiga
        result = centroids_next - centroids
        # print("result",result)
        # for i in range(k):
        #     print(np.linalg.norm(result[i]))
        max_list = [float(np.linalg.norm(result[i])) for i in range(k)]  # 求出c(j,t+1)-c(j,t)的值
        max_center_value = max(max_list)  # 取其中最大值
        # print('max_center_value',max_center_value)
        if max_center_value < omiga:  # 判断max<omiga,是则退出循环
            # print("Success at No.%d times" % iter)
            break
        centroids = centroids_next
    return centroids, subCenter


if __name__ == "__main__":
    #data = np.random.rand(100,3)
    data = np.load('matrix_keywords_tfidf_1000m_5.npy')
    #data = np.array(labeling(data.tolist()))
    np.random.shuffle(data)
    center_list, cluster_category = (Skmeanspp(data,5,0.0000001,1000))

    #print(centroid)
    #print(subCenter)
    #print(center_list)
    #print(cluster_category)

    center_list = np.array(center_list)
    cluster_category = np.array(cluster_category)
    print(np.shape(center_list))
    print(np.shape(cluster_category))
#    print(center_list)
#    print(cluster_category)

    c0 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    category = []
    print(category)
    for sample in cluster_category:
        if sample[0] == 0:
            c0.append(sample)
        elif sample[0] == 1:
            c1.append(sample)
        elif sample[0] == 2:
            c2.append(sample)
        elif sample[0] == 3:
            c3.append(sample)
        elif sample[0] == 4:
            c4.append(sample)

    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    c4 = np.array(c4)
    category.append(c0)
    category.append(c1)
    category.append(c2)
    category.append(c3)
    category.append(c4)

    category = np.array(category)
    print(np.shape(category))

    labels = []
    
    #print(category)

    for i in category:
        #print(type(i))
        a,b,c,d,e = 0,0,0,0,0
        for vec in i:
            #print(vec)
            value = int(vec[-1])
            if 0 <= value < 200:
                #print("Art", value)
                a += 1
            elif 200 <= value < 400:
                #print("Enviornment", value)
                b += 1
            elif 400 <= value < 600:
                #print("Agriculture", value)
                c += 1
            elif 600 <= value < 800:
                #print("Economy", value)
                d += 1
            elif 800 <= value < 1000:
                #print("Politics", value)
                e += 1
        labels.append([a, b, c, d, e])

    print(labels)
