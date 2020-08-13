import departing_and_reducing_stopwords as DARS
import numpy as np
from TFIDF import TextVectorizer as TV
from BSKM import *
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
#from compiler.ast import flatten
def getfiles(path):
    #filenames = os.listdir(r'E:\temp\program\python\git_2020_dachuang\dataset_mc\dataset_mc\bishe200m5')
    filenames = os.listdir(path)
    #print(filenames)
    return filenames
#getfiles(r'E:\temp\program\python\git_2020_dachuang\dataset_mc\dataset_mc\bishe200m5')

def preprocessing(filenames):
    for i in filenames:
        time_start = time.time()
        prefix = 'E:/temp/program/python/git_2020_dachuang/experiment/'
        output_file_name = prefix + i
        input_prefix = 'E:/temp/program/python/git_2020_dachuang/dataset_mc/dataset_mc/bishe200m5/'
        input_text = input_prefix + i
        tv = TV('E:\\temp\\program\\python\\git_2020_dachuang\\dataset_mc\\dataset_mc\\bishe200m5corpus_result.txt',input_text,'stopwords.txt','corpus_output.txt',output_file_name,max_df=0.15,min_df=0.0002)
        tv.init_corpus_and_text()
        tv.fitting()
        tv.transforming()
        wordlist = tv.getWordList()
        weightlist = tv.getWeightList()
        dict_ = tv.getDict()
        file1 = prefix + i + 'file1.txt'
        file2 = prefix + i + 'file2.txt'
        file3 = prefix + i + 'file_reduce.txt'
        tv.writeInFile(file1)
        tv.writeTfidfInFileSorted(file2)
        textvector,feature_name,feature_tfidf = tv.getTextVector()

        dim = len(feature_name)
        #print(dim)
        size = dim
        w = 1.4
        c1=c2=1.2
        gama = 0.9
        theta = 5
        max_vel = 1
        iter_num = 100
        gsbpso = GSBPSO(dim,size,iter_num,max_vel,theta,feature_tfidf,gama,c1=c1,c2=c2,w=w)

        fitness_value_list,best_position = gsbpso.update()
        print(best_position)
        print(fitness_value_list[-1])
        selected_feature = []
        for i in range(len(best_position)):
            if best_position[i] == 1:
                selected_feature.append(i)
        print(selected_feature)
        list_feature = []    
        for i in selected_feature:
            print(feature_name[i])
            list_feature.append(feature_name[i])
        
        with open(file3,'w',encoding='utf-8') as f3:
            for i in list_feature:
                f3.write(i)
                f3.write(' ')

        time_end = time.time()
        print('time cost:',time_end-time_start,'s')

def labeling(x):

    for i in range(len(x)):
        x[i].append(int(i))
    return np.array(x)

def getAllFeaturesAssemble(reduce_filenames):
    t1 = time.time()
    prefix = 'E:/temp/program/python/git_2020_dachuang/experiment/'
    
    features_list = []
    for i in reduce_filenames:
        file_name = prefix + i
        each_featrue = []
        tmp_prefix = i.split('.')[1]
        if tmp_prefix == 'txtfile_reduce':
            with open(file_name,'r',encoding='utf-8') as f:
                #f.readline()
                each_featrue.append(f.readline())
                print(each_featrue)
            features_list.append(each_featrue)
        else:
            continue
        t2 = time.time()
        print('cost:',t2-t1,'s')
    return features_list


if __name__ == "__main__":
    #这两步用于对文本数据进行GSBPSO处理 得到约简后的文本特征集
    #------------------------------------------------------------------
    #filenames = getfiles(r'E:\temp\program\python\git_2020_dachuang\dataset_mc\dataset_mc\bishe200m5'
    #preprocessing(filenames)
    #------------------------------------------------------------------

    #下面步骤用于提取处理后的文本特征集合
    #------------------------------------------------------------------
    #reduce_filenames = getfiles(r'E:\temp\program\python\git_2020_dachuang\experiment')
    #features_list = getAllFeaturesAssemble(reduce_filenames)
    #features_list = np.array(features_list)
    #np.save('features_list.npy',features_list)
    #print(features_list)
    #print(features_list)
    #------------------------------------------------------------------
    
    #下面步骤用于将特征集合编码 用于下一步的聚类
    features_list_tmp = np.load('features_list.npy')
    features_list = []
    features_list.extend([x[0] for x in features_list_tmp])

    tfidf_vec = TfidfVectorizer()
    tfidf_mat = tfidf_vec.fit_transform(features_list)
    #print(tfidf_mat.toarray())
    #print(np.shape(tfidf_mat.toarray()))

    tfidf_mat_array = tfidf_mat.toarray()
    tfidf_mat_array = labeling(tfidf_mat_array.tolist())
    #bskm = BSKM(tfidf_mat_array,5,300,0.0001)
    #bskm.fit()
    #print(bskm.centers)
    #centers = bskm.centers
    #clf = bskm.clf
    #------------------------------------------------------------------
    #保存分好的簇以及各个样本归属情况
    #------------------------------------------------------------------
    #with open('centers.txt','wb') as f1:
    #    pickle.dump(centers,f1)
    
    #with open('clf.txt','wb') as f2:
        #f2.write(str(clf))
    #    pickle.dump(clf,f2)

    bskm = BSKM(tfidf_mat_array,5,300,0.0001)
    bskm.fit()
    print(bskm.centers)
    print(bskm.clf_label)
    centers = bskm.centers
    clf = bskm.clf
    clf_label = bskm.clf_label
    #------------------------------------------------------------------
    #保存分好的簇以及各个样本归属情况
    #------------------------------------------------------------------
    with open('centers.txt','wb') as f1:
        pickle.dump(centers,f1)
    
    with open('clf.txt','wb') as f2:
        #f2.write(str(clf))
        pickle.dump(clf,f2)

    with open('clf_labeled.txt','wb') as f3:
        pickle.dump(clf_label,f3)

    with open('centers.txt','rb+') as f1:
        centers_load = pickle.load(f1)

    with open('clf.txt','rb+') as f2:
        clf_load = pickle.load(f2)
        
    with open('clf_labeled.txt','rb+') as f3:
        clf_labeled_load = pickle.load(f3)
        
    #print(type(centers_load))
    #print(centers_load)
    #print(type(clf_load))
    #print(clf_load)
    #np.save('centers.npy',centers) 导致保存的不是dict 弃用
    #np.save('clf.npy',clf)


    #------------------------------------------------------------------
    #centers_load = np.load('centers.npy',allow_pickle=True)
    #clf_load = np.load('clf.npy',allow_pickle=True)
    #print('dict = ',centers_load)
    #print('clf = ',clf_load)
    #------------------------------------------------------------------

    #print(label)
    
    #print(centers)
    #print(clf)
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #print(clf_load)

    
    #print(len(tfidf_mat_array.tolist()))

    
    
