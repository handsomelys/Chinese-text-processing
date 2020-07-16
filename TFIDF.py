from departing_and_reducing_stopwords import Departing_word as dwd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction
import numpy as np
from pprint import  pprint

'''
#np.set_printoptions(threshold=np.inf)

word_raw = dwd('stopwords.txt', 'verse.txt', 'out.txt')  # 测试文本
word_raw.departing()
words = word_raw.get_words()


string = " ".join(words)
words_text = []
words_text.append(string)


corpus_raw = dwd('stopwords.txt', 'corpus.txt', 'corpus_out.txt')  # 自制小语料库测试
corpus_raw.departing()
corpus = corpus_raw.get_words()

# 构建tfidf模型

#print(voc_list)
tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",max_df=0.15,min_df=0.002)
#tfidf_model = tfidf_vec.fit(corpus)
#words_text.extend(corpus_text)
#print(words_text)
#tfidf_matrix = tfidf_vec.fit_transform(words)

tfidf_vec.fit(corpus)
tfidf_matrix = tfidf_vec.transform(words)

#print(tfidf_vec.inverse_transform(tfidf_matrix))

for i in tfidf_matrix.toarray():
    print(i)
print(tfidf_matrix.toarray())

# print(tfidf_vec.get_feature_names())
# 词表
wordlist = tfidf_vec.get_feature_names()
#print(tfidf_vec.vocabulary_)
#print(tfidf_matrix.toarray())
print(type(wordlist))
#print(wordlist)
#print(len(wordlist))
# 权值表
weightlist = tfidf_matrix.toarray()
#print(len(weightlist))
print(type(weightlist))
# tf-idf矩阵中 元素a[i][j]表示j词在i类文本中的tfidf权重
# 写入文本
with open("tfidf_word.txt", 'w', encoding='utf-8')as f:
    for i in range(len(weightlist)):
        f.write("第"+str(i)+"段文本："+"\n")
        # print("------第",i,'段文本词语的tfidf权重')
        for j in range(len(wordlist)):
            #print(wordlist[j],weightlist[i][j])
            f.write(wordlist[j]+"  "+str(weightlist[i][j])+"\n")


dict_final = {}
# count2=0
for i in range(len(weightlist)):
    for j in range(len(wordlist)):
        
        try:
            dict_final[wordlist[j]]+=weightlist[i][j]
        except:
            #print(wordlist[j])
            # count2+=1
            dict_final[wordlist[j]]=0
            dict_final[wordlist[j]] += weightlist[i][j]
        #print(wordlist[j],weightlist[i][j])
        #print('?')
        #dict_final[wordlist[j]] = weightlist[i][j]

dict_final_sort = sorted(dict_final.items(), key=lambda x: x[1], reverse=True)

#print(dict_final_sort)
#print(dict(dict_final_sort))
with open("main_txt.txt",'w')as f:
    for i in list(dict_final_sort):
        every=str(i)
        # print(every)
        f.write(every+'\n')
'''
class TextVectorizer:
    def __init__(self,corpus_name,text_name,stopwords,corpus_output,text_output,max_df,min_df):
        self.tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",max_df=max_df,min_df=min_df)
        self.corpus_name = corpus_name
        self.text_name = text_name
        self.stopwords = stopwords
        self.corpus_output = corpus_output
        self.text_output = text_output

    def init_corpus_and_text(self):
        corpus_raw = dwd(self.stopwords, self.corpus_name, self.corpus_output)  # 自制小语料库测试
        corpus_raw.departing()
        corpus = corpus_raw.get_words()
        self.corpus = corpus

        word_raw = dwd(self.stopwords, self.text_name, self.text_output)  # 测试文本
        word_raw.departing()
        words = word_raw.get_words()
        self.text = words

    def fitting(self):
        self.tfidf_vec.fit(self.corpus)

    def transforming(self):
        self.tfidf_matrix = self.tfidf_vec.transform(self.text)
    
    def getWordList(self):
        return self.tfidf_vec.get_feature_names()

    def getWeightList(self):
        return self.tfidf_matrix.toarray()

    def writeInFile(self,filename):
        weightlist = self.getWeightList()
        wordlist = self.getWordList()
        with open(filename, 'w', encoding='utf-8')as f:
            for i in range(len(weightlist)):
                f.write("第"+str(i)+"段文本："+"\n")
                # print("------第",i,'段文本词语的tfidf权重')
                for j in range(len(wordlist)):
                    #print(wordlist[j],weightlist[i][j])
                    f.write(wordlist[j]+"  "+str(weightlist[i][j])+"\n")

    def getDict(self):
        dict_final = {}
        weightlist = self.getWeightList()
        wordlist = self.getWordList()
        for i in range(len(weightlist)):
            for j in range(len(wordlist)):
                
                try:
                    dict_final[wordlist[j]]+=weightlist[i][j]
                except:
                    dict_final[wordlist[j]]=0
                    dict_final[wordlist[j]] += weightlist[i][j]
        dict_final_sort = sorted(dict_final.items(), key=lambda x: x[1], reverse=True)
        return dict_final_sort

    def writeTfidfInFileSorted(self,filename):
        dict_ = self.getDict()
        with open(filename,'w')as f:
            for i in list(dict_):
                every=str(i)
                # print(every)
                f.write(every+'\n')

    def getTextVector(self):
        feature_name = []
        feature_tfidf = []
        textvector = []
        dict_ = self.getDict()
        
        for i in dict_:
            #tuple_ = ()
            if i[1] > 0:
                feature_name.append(i[0])
                feature_tfidf.append(i[1])
                
                #tuple_ = (i)
                
                textvector.append(i)
        return textvector,feature_name,feature_tfidf


'''
corpus_raw = dwd('stopwords.txt', 'corpus.txt', 'corpus_out.txt')  # 自制小语料库测试
corpus_raw.departing()
corpus = corpus_raw.get_words()

word_raw = dwd('stopwords.txt', 'verse.txt', 'out.txt')  # 测试文本
word_raw.departing()
words = word_raw.get_words()
'''
if __name__ == "__main__":
    '''
    tv = TextVectorizer('corpus.txt','verse.txt','stopwords.txt','corpus_output.txt','text_output.txt',max_df=0.15,min_df=0.0002)
    tv.init_corpus_and_text()
    tv.fitting()
    tv.transforming()
    wordlist = tv.getWordList()
    weightlist = tv.getWeightList()
    dict_ = tv.getDict()
    tv.writeInFile("file1.txt")
    tv.writeTfidfInFileSorted("file2.txt")
    textvector,feature_name,feature_tfidf = tv.getTextVector()
    print(textvector)
    print(feature_name)
    print(feature_tfidf)
    #print(wordlist)
    #for i in weightlist:
        #print(i)
    #print(dict_)
    #print(feature_name)
    #print(len(feature_name))
    '''
    pass