# coding:UTF-8
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint


def load_data(file_path):
    '''导入数据
    input:  file_path(string):文件的存储位置
    output: data(mat):数据
    '''
    f = open(file_path)
    data = []
    for line in f.readlines():
        row = []  # 记录每一行
        lines = line.strip().split("\t")
        for x in lines:
            row.append(float(x))  # 将文本中的特征转换成浮点数
        data.append(row)
    f.close()
    return np.mat(data)

def randCent(data, k):
    '''随机初始化聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
    output: centroids(mat):聚类中心
    '''
    n = np.shape(data)[1]  # 属性的个数
    centroids = np.mat(np.zeros((k, n)))  # 初始化k个聚类中心
    for j in range(n):  # 初始化聚类中心每一维的坐标
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        # 在最大值和最小值之间随机初始化
        centroids[:, j] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangeJ
    return centroids

def Skmeans(data, k, omiga, max_iter):
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
    centroids = normalize(randCent(data[:, :-1], k))  # 随机初始化聚类中心
    data[:, :-1] = normalize(data[:, :-1])  # 标准化数据
    for iter in range(max_iter):
        # centroids = normalize(randCent(data, k))  # 随机初始化聚类中心
        for i in range(m):  # 更新隶属度
            max_index = 0
            max_value = -np.inf
            for j in range(k):  # 遍历全部聚类中心，求出最大值即属于哪个聚类中心
                if (data[i, :-1] * centroids[j].T)[0, 0] > max_value:
                    max_value = (data[i, :-1] * centroids[j].T)[0, 0]  # 求样本和聚类中心的内积，得到最大值
                    max_index = j  # 得到最大值的下标即聚类中心的编号
            lst = [max_index]  # 将样本聚类标签和样本一同写入subCenter矩阵中
            lst.extend(data[i].tolist()[0])
            subCenter[i,] = np.mat(lst)
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


def bin_skmeans(data_set, k, omiga, max_iter):
    # labels = data_set[:, -1] # 每个文本矩阵的标签
    # data_set = data_set[:,:-1] # 文本矩阵
    # m, n = np.shape(data_set)  # 得到样本数据的个数和维数
    centroid_0 = normalize(np.mean(data_set[:, :-1], axis=0)[0])
    center_list = [centroid_0]
    data_set[:, :-1] = normalize(data_set[:, :-1])
    cluster_category = [[]]  # 用于存储不同的簇
    cluster_category[0] = data_set
    while len(cluster_category) < k:
        max_sce = 0  # 存储SCE最大簇的值
        max_sce_id = 0  # 最大簇下标
        for center_id, cluster in enumerate(cluster_category):  # 求出sce最大的簇
            temp_sce = 0
            for x in cluster[:, :-1]:  # 计算当前簇的SCE
                temp_sce += (1 - x * center_list[center_id].T)
            if temp_sce > max_sce:  # 是否是SCE最大簇
                max_sce = temp_sce
                max_sce_id = center_id
        # 得到最大簇后进行球面聚类，得到聚类中心centroid_temp 和 包含标签和样本数据的subcenter_temp
        test_num = 10
        centroid_temp, subceter_temp = [[] for _ in range(test_num)], [[] for _ in range(test_num)]
        for t in range(test_num):  # 设置SKM测试次数为10
            centroid_temp[t], subceter_temp[t] = Skmeans(data=cluster_category[max_sce_id], k=2, omiga=omiga,
                                                         max_iter=max_iter)
        cluster_category.pop(max_sce_id)  # 删除原有的簇
        center_list.pop(max_sce_id)  # 删除原有的簇的中心
        # 选取其中SCE最小的一组作为分裂结果
        min_sce = np.inf  # 存储SCE最大簇的值
        min_sce_id = 0  # 最大簇下标
        for cid, subctr in enumerate(subceter_temp):  # 求出sce最大的簇
            temp_sce = 0
            # for x in subctr[cid, 1:-1][0]:  # 计算当前簇的SCE
            #     temp_sce += (1 - x * centroid_temp[cid][0].T)
            # for x in subctr[cid, 1:-1][0]:
            #     temp_sce += (1 - x * centroid_temp[cid][1].T)
            for x in range(len(subctr)):  # 计算当前簇的SCE
                temp_sce += (1 - subctr[x, 1:-1][0] * centroid_temp[cid][0].T)
            for x in range(len(subctr)):
                temp_sce += (1 - subctr[x, 1:-1][0] * centroid_temp[cid][1].T)
            if temp_sce < min_sce:  # 是否是SCE最小簇
                min_sce = temp_sce
                min_sce_id = cid
        # print(min_sce, min_sce_id)
        # print("=="*30) # 对得到的subceter_temp进行分割
        subceter_tmp = subceter_temp[min_sce_id]
        cluster_temp = [[], []]  # 将聚类后的簇分割成两个簇
        for id in range(len(subceter_tmp)):
            if subceter_tmp[id, 0] == 0:
                cluster_temp[0].append(subceter_tmp[id, 1:].tolist()[0])
            else:
                cluster_temp[1].append(subceter_tmp[id, 1:].tolist()[0])
        # print("="*30)
        # print(cluster_temp[0])
        # print(len(cluster_temp[0]))
        # print(cluster_temp[1])
        # print(len(cluster_temp[1]))
        cluster_temp[0] = np.mat(cluster_temp[0])  # 第0号簇
        cluster_temp[1] = np.mat(cluster_temp[1])  # 第1号簇
        # print(len(cluster_temp[0]),len(cluster_temp[1]))
        cluster_category.append(cluster_temp[0])  # 将分裂结果的两个簇加入簇表
        cluster_category.append(cluster_temp[1])
        center_list.append(centroid_temp[min_sce_id][0])  # 将分裂结果的两个簇的中心加入簇中心表
        center_list.append(centroid_temp[min_sce_id][1])
        print("已分出%d个簇" % len(cluster_category))
    # print(center_list)
    # for item in cluster_category:
    #     print(item)
    return center_list, cluster_category


def bin_show(center_list, cluster_category):
    color_box = ['#0000FE', '#000000', '#FE0000', '#99CC00', '#FF00FF']
    print("=================cluster_category===================")
    for id, line in enumerate(cluster_category):
        X, Y = [item[0, 0] for item in line], [item[0, 1] for item in line]
        print("=============No.%d============" % id)
        # for item in line:
        #     X.append(item[0,0])
        #     Y.append(item[0,1])
        # print(item)
        plt.scatter(X, Y, 50, color=color_box[id], marker='.', linewidth=2, alpha=0.8)
    center_X, center_Y = [item[0, 0] for item in center_list], [item[0, 1] for item in center_list]
    print("=================center_list===================")
    # for item in center_list:
    # print(item)
    #     # print(item[0,0],item[0,1])
    #     center_X.append(item[0,0])
    #     center_Y.append(item[0,1])
    plt.scatter(center_X, center_Y, 150, color='#00FF00', marker='+', linewidth=2, alpha=0.8)
    plt.grid(color='#95a5a6')
    plt.show()


def normalize(input_data):
    for i in range(len(input_data)):
        mod = np.linalg.norm(input_data[i])
        input_data[i] = input_data[i] / mod
    # print(input_data)
    return input_data


def save_result(file_name, source):
    '''保存source中的结果到file_name文件中
    input:  file_name(string):文件名
            source(mat):需要保存的数据
    output:
    '''
    m, n = np.shape(source)
    f = open(file_name, "a+")
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()


def show(num, center, subcenter, row_data):
    data = row_data.tolist()
    show_data = [[] for i in range(num)]
    num_count = 0
    while num_count < num:
        for index, item in enumerate(subcenter.tolist()):
            if item[0] == num_count:
                show_data[num_count].append(data[index])
        num_count += 1
    color_box = ['#0000FE', '#000000', '#FE0000', '#99CC01']
    for id, line in enumerate(show_data):
        X, Y = [], []
        for item in line:
            X.append(item[0])
            Y.append(item[1])
        plt.scatter(X, Y, 50, color=color_box[id], marker='.', linewidth=2, alpha=0.8)
    center_X, center_Y = [], []
    for item in center.tolist():
        center_X.append(item[0])
        center_Y.append(item[1])
        plt.scatter(center_X, center_Y, 150, color='#00FF00', marker='+', linewidth=2, alpha=0.8)
    plt.grid(color='#95a5a6')
    plt.show()


def test_useable(cluster_category):
    max_label = []
    max_value = []
    for every_cluster in cluster_category:
        a, b, c, d, e = 0, 0, 0, 0, 0
        for key in range(len(every_cluster[:, -1])):
            value = int(every_cluster[key, -1])
            if 0 <= value < 200:
                a += 1
            elif 200 <= value < 400:
                b += 1
            elif 400 <= value < 600:
                c += 1
            elif 600 <= value < 800:
                d += 1
            elif 800 <= value < 1000:
                e += 1
        temp = [a, b, c, d, e]
        max_value.append(max(temp))
        max_index = temp.index(max(temp))
        max_label.append(max_index)
    if len(set(max_label))==5:
        if min(max_value)<100:
            return False
        else:
            return True
    else:
        return False


def main_socket():
    # data = np.mat(np.load('test_matrix.npy'))
    # file_path = "data.txt"
    # # 1、导入数据
    # print("---------- 1.load data ------------")
    # data= np.mat(np.random.rand(100,2))
    # # 2、随机初始化k个聚类中心
    # print("---------- 2.random center ------------")
    # centroids = randCent(data, k)
    # # 3、聚类计算
    # print("---------- 3.kmeans ------------")
    # subCenter = kmeans(data, k, centroids)
    # # print(subCenter)
    # # 4、保存聚类中心
    # print("---------- 4.save centroids ------------")
    # save_result("center.txt", centroids)
    # # 5、展示分类结果
    # print("---------- 5.show result ------------")
    # show(k, centroids, subCenter, data)
    # ---------------------------------------------------------
    # centroids, subCenter = Skmeans(data, k, omiga=0.1, max_iter=1000)
    # show(k, centroids, subCenter, data)
    k = 5  # 聚类中心的个数
    # file_path = "data2.txt"
    # data = load_data(file_path)
    data = np.mat(np.load("matrix_keywords_tfidf_1000m_5.npy"))
    # data = np.append(data,data,axis=0)
    np.random.shuffle(data)
    # save_result("matrix_keywords_tfidf_plus.txt",data)
    failed = 1
    while True:
        center_list, cluster_category = bin_skmeans(data_set=data, k=k, omiga=0.0000001, max_iter=1000)
        if test_useable(cluster_category):
            print("在第%d次聚类成功" % failed)
            break
        else:
            print("第%d次聚类失败" % failed)
        failed += 1

    print("center_list", "==" * 30)
    for i in center_list:
        save_result("center_list1000m5_3.txt", i)
        for j in i:
            print(j, end=" ")
        # f.write("\r")
        print("\r")
    print("cluster", "==" * 30)
    clst = []
    labels = []
    for cl in cluster_category:
        clster = 0
        a, b, c, d, e = 0, 0, 0, 0, 0
        save_result("cluster_category1000m5_3.txt", cl)
        with open("cluster_category1000m5_3.txt", "a+")as f:
            f.write("\n")
        for key in range(len(cl[:, -1])):
            value = int(cl[key, -1])
            if 0 <= value < 200:
                print("Art", value)
                a += 1
            elif 200 <= value < 400:
                print("Enviornment", value)
                b += 1
            elif 400 <= value < 600:
                print("Agriculture", value)
                c += 1
            elif 600 <= value < 800:
                print("Economy", value)
                d += 1
            elif 800 <= value < 1000:
                print("Politics", value)
                e += 1
            clster += 1
        clst.append(clster)
        labels.append([a, b, c, d, e])
        print("==" * 30)
    print(clst)
    print("=" * 30)
    for label in labels:
        print(label)
    # # bin_show(center_list, cluster_category)
    # 经检验四维数据集聚类没有报错
    # print('center_list'+"=" * 40)
    # print(center_list)
    # print('cluster_category'+"="*40)
    # print(cluster_category)
    # 将data.txt中数据的维度翻倍
    # with open("data.txt","r")as fr:
    #     with open("data2.txt","w")as fw:
    #         for line in fr.readlines():
    #             lines = line.strip().split("\t")
    #             fw.write("\t".join(lines+lines))
    #             fw.write("\r")


if __name__ == "__main__":
    main_socket()
    '''
    0 - 39:
    艺术
    40 - 79:
    环境
    80 - 119:
    农业
    120 - 159:
    经济
    160 - 199:
    政治
    
    聚类各簇的个数：
            38 , 43 , 43 , 24 , 52
    最多    艺术  艺术 政治  环境 环境
    '''
