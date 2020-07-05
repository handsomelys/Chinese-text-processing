from departing_and_reducing_stopwords import Departing_word as Dw
import math
import numpy as np
class Build_word_bag:
    def __init__(self):
        self.vocabSet = set([])
        self.returnVec = None
        self.vocabList = []
    def createVocabList(self,dataSet):
        self.vocabSet = set([])

        for word in dataSet:
            if word not in self.vocabSet:
                self.vocabSet.add(word)
        return list(self.vocabSet)

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
        #print(self.vocabList)
        #print('\n\n\n')
        #print(dataSet)
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
    #print(words)
    #print(vocabList)
    vocabList.sort()
    #print(dataSet)
    #print(vocabList)
    word_vec = build_word_bag.bagofWords2Vec(vocabList,words)
    #print(word_vec)

    tfidf_model = TFIDF(vocabList)
    tfidf_model.getTF_IDF(words_raw1,'TF-IDF.txt')
    tfidf_list = np.loadtxt('TF-IDF.txt')
    #print(word_vec)
    #print(vocabList)
    #print(vocabList.index('人权'))
    print(tfidf_list)
if __name__ == '__main__':
    main()