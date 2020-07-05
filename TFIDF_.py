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
        return self.returnVec

class TFIDF:
    def __init__(self,vocabList):
        super().__init__()
        self.words_cnt = 0
        self.file_cnt = 0
        self.all_words_cnt = 0
        self.vocabList = vocabList
        self.bwb = Build_word_bag()
        self.tfidf_list = []
    
    def wordsCount(self,dataSet):
        words_cnt = 0
        for document in dataSet:
            words_cnt += len(document)
        return words_cnt

    def wordInFileCount(self,word,word_list):
        file_cnt = 0
        for i in word_list:
            for j in i:
                if word == j:
                    self.file_cnt = self.file_cnt + 1
                else:
                    continue
        return file_cnt

    def getTF_IDF(self,dataSet,outputFile):
        self.all_words_cnt = self.wordsCount(dataSet)
        temp_file_cnt = len(dataSet)
        
        file_output = open(outputFile,'w')
        for line in dataSet:
            words_bag = self.bwb.bagofWords2Vec(self.vocabList,line)
            line_words_cnt = line.count(' ')
            
            self.tfidf_list = [0]*len(self.vocabList)
            word_temp = []
            word_temp.extend(line.split())
            for word in word_temp:                
                word_in_file_cnt = self.wordInFileCount(word,dataSet)
                word_cnt = words_bag[self.vocabList.index(word)]
                tf = float(word_cnt)/line_words_cnt
                idf = math.log(float(temp_file_cnt)/(word_in_file_cnt + 1))
                tfidf = tf*idf
                self.tfidf_list[self.vocabList.index(word)] = tfidf
            file_output.write('\t'.join(map(str,self.tfidf_list)))
            file_output.write('\n')
        file_output.close()

    def get_tfidf_list(self):
        return self.tfidf_list

def reduce_raw(vocabList,dataSet):
    dict_for_words = {}
    all_word_cnt = len(dataSet)
    for i in vocabList:
        dict_for_words[i] = 0
    #print(dict_for_words)
    for line in dataSet:
        word_temp = []
        word_temp.extend(line.split())
        for word in word_temp:
            dict_for_words[word] = dict_for_words[word] + 1
    #print(dict_for_words)
    #print(all_word_cnt)
    #print(float(1/all_word_cnt))    #可能要除去小于0.00042
    return dict_for_words
def reduce_word_(vocabList,words):
    reduce_word = vocabList
    dict_for_words = reduce_raw(reduce_word,words)
    reduce_name = []
    for key in dict_for_words:
        if dict_for_words[key] < float(0.00042*len(words)) or dict_for_words[key] > float(0.15*len(words)):
            reduce_name.append(key)
    
    reduce_name = list(set(reduce_name))
    '''
    for i in reduce_name:
        while i in reduce_word:
            reduce_word.remove(i)
        '''
    return reduce_name
def reduce_txt(vocabList,words):
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
    #准备进行第一次粗选
    reduce_txt(vocabList,words)
    #reduce_word = reduce_word_(vocabList,words)
    reduce_data_raw = Dw('stopwords.txt','reduce_txt.txt','output_reduce.txt')
    reduce_data_raw.departing()
    reduce_data_raw1 = reduce_data_raw.get_words()
    reduce_word = []
    for i in reduce_data_raw1:
        reduce_word.extend(i.split())
    
    reduce_dataSet = set(reduce_word)
    reduce_vocabList = build_word_bag.createVocabList(reduce_dataSet)
    reduce_vocabList.sort()

    word_vec = build_word_bag.bagofWords2Vec(reduce_vocabList,reduce_word)

    tfidf_model = TFIDF(reduce_vocabList)
    tfidf_model.getTF_IDF(reduce_data_raw1,'TF-IDF.txt')
    tfidf_list = np.loadtxt('TF-IDF.txt')
    print(tfidf_list)

if __name__ == '__main__':
    main()