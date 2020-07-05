from departing_and_reducing_stopwords import Departing_word as Dw
import math
import numpy as np
class Build_word_bag:
    def __init__(self):
        self.vocabSet = set([])
        self.returnVec = None
        self.vocabList = []
    #创建不重复的词条列表
    def createVocabList(self,dataSet):
        self.vocabSet = set([])

        for word in dataSet:
            if word not in self.vocabSet:
                self.vocabSet.add(word)
        return list(self.vocabSet)
    #将文本转化为词袋模型
    def bagofWords2Vec(self,vocabList,inputSet):
        self.returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                self.returnVec[vocabList.index(word)] += 1 
            else:
                #print("this word: (%s) not in " %(word))
                pass
        return self.returnVec   #返回得到的词向量

class TFIDF:
    def __init__(self,vocabList):
        super().__init__()
        self.words_cnt = 0
        self.file_cnt = 0
        self.all_words_cnt = 0
        self.vocabList = vocabList
        self.bwb = Build_word_bag()
        self.tfidf_list = []
    
    def wordsCount(self,dataSet):   #统计词频
        words_cnt = 0
        for document in dataSet:
            words_cnt += len(document)
        return words_cnt

    def wordInFileCount(self,word,word_list):   #统计文档频率
        file_cnt = 0
        for i in word_list:
            for j in i:
                if word == j:
                    self.file_cnt = self.file_cnt + 1
                else:
                    continue
        return file_cnt

    def getTF_IDF(self,dataSet,outputFile): #计算TF_IDF

        self.all_words_cnt = self.wordsCount(dataSet)
        
        temp_file_cnt = len(dataSet)    #获得总的词量
        
        file_output = open(outputFile,'w')  #将得到的词向量写进文件
        for line in dataSet:    #每一行文本统计
            words_bag = self.bwb.bagofWords2Vec(self.vocabList,line)
            line_words_cnt = line.count(' ')    #通过计算一行文本里的空格 得到这个文本的词数
            
            self.tfidf_list = [0]*len(self.vocabList)   #初始化list
            word_temp = []  #将每行文本的词放进这个临时变量
            word_temp.extend(line.split())
            for word in word_temp:                
                word_in_file_cnt = self.wordInFileCount(word,dataSet)   #统计文档频率
                word_cnt = words_bag[self.vocabList.index(word)]    #统计词频
                tf = float(word_cnt)/line_words_cnt #计算tf
                idf = math.log(float(temp_file_cnt)/(word_in_file_cnt + 1)) #计算idf
                tfidf = tf*idf
                self.tfidf_list[self.vocabList.index(word)] = tfidf #赋值
            file_output.write('\t'.join(map(str,self.tfidf_list)))  #把向量写进文件里
            file_output.write('\n')
        file_output.close()

    def get_tfidf_list(self):   #getter
        return self.tfidf_list

def reduce_raw(vocabList,dataSet):  #获得一个键值对为 键：词，值：频数的字典
    dict_for_words = {}
    all_word_cnt = len(dataSet)
    for i in vocabList:
        dict_for_words[i] = 0
    for line in dataSet:
        word_temp = []
        word_temp.extend(line.split())
        for word in word_temp:
            dict_for_words[word] = dict_for_words[word] + 1 #统计
    return dict_for_words

def reduce_word_(vocabList,words):  #去除TF<0.042%&&TF>0.15的特征词
    reduce_word = vocabList
    dict_for_words = reduce_raw(reduce_word,words)
    reduce_name = []
    for key in dict_for_words:  #dict_for_words[key]为TF
        if dict_for_words[key] < float(0.00042*len(words)) or dict_for_words[key] > float(0.15*len(words)):
            reduce_name.append(key)    
    reduce_name = list(set(reduce_name))

    return reduce_name

def reduce_txt(vocabList,words):    #将约简后的文本数据写到文件里
    f = open('out.txt','r',encoding='utf-8')
    reduce_name = reduce_word_(vocabList,words)
    lst = []
    for line in f:
        for word in reduce_name:
            line = line.replace(word,'')
        lst.append(line)
    f.close()
    f = open('reduce_txt.txt','w',encoding='utf-8')
    for line in lst:
        f.write(line)
    f.close()

def main():
    #先得到一个词表
    build_word_bag = Build_word_bag()
    word_raw = Dw('stopwords.txt','verse.txt','out.txt')
    word_raw.departing()
    words_raw1 = word_raw.get_words()
    words = []
    for i in words_raw1:
        words.extend(i.split())
    dataSet = set(words)
    vocabList = build_word_bag.createVocabList(dataSet)
    vocabList.sort()

    #进行第一次粗选 去除TF<0.042%&&TF>0.15的特征词
    reduce_txt(vocabList,words)
    reduce_data_raw = Dw('stopwords.txt','reduce_txt.txt','output_reduce.txt')
    reduce_data_raw.departing()
    reduce_data_raw1 = reduce_data_raw.get_words()
    reduce_word = []

    for i in reduce_data_raw1:  #分割文本 即将一个词语作为一个元素放到list中
        reduce_word.extend(i.split())
    
    reduce_dataSet = set(reduce_word)   #去重
    reduce_vocabList = build_word_bag.createVocabList(reduce_dataSet)   #创建词表
    reduce_vocabList.sort() #排序

    word_vec = build_word_bag.bagofWords2Vec(reduce_vocabList,reduce_word)  #得到词向量

    tfidf_model = TFIDF(reduce_vocabList)
    tfidf_model.getTF_IDF(reduce_data_raw1,'TF-IDF.txt')    #计算得到各个特征词的TF_IDF
    tfidf_list = np.loadtxt('TF-IDF.txt')
   # print(tfidf_list)

if __name__ == '__main__':
    main()