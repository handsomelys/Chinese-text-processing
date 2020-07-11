from departing_and_reducing_stopwords import Departing_word as dwd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction
import numpy as np
from pprint import  pprint


#np.set_printoptions(threshold=np.inf)

word_raw = dwd('stopwords.txt', 'verse3.txt', 'out.txt')  # 测试文本
word_raw.departing()
words = word_raw.get_words()

string = " ".join(words)
words_text = []
words_text.append(string)
#print(words_text)

corpus_raw = dwd('stopwords.txt', 'corpus.txt', 'corpus_out.txt')  # 自制小语料库测试
corpus_raw.departing()
corpus = corpus_raw.get_words()

string = " ".join(corpus)
corpus_text = []
corpus_text.append(string)

# 构建tfidf模型

tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",max_df=0.8,min_df=0.002)

#tfidf_model = tfidf_vec.fit(corpus)
words_text.extend(corpus_text)
#print(words_text)
tfidf_matrix = tfidf_vec.fit_transform(words_text)
'''
for i in tfidf_matrix.toarray():
    print(i)
print(tfidf_matrix.toarray())
'''
# print(tfidf_vec.get_feature_names())
# 词表
wordlist = tfidf_vec.get_feature_names()
print(wordlist)
#print(len(wordlist))
# 权值表
weightlist = tfidf_matrix.toarray()
#print(len(weightlist))

# tf-idf矩阵中 元素a[i][j]表示j词在i类文本中的tfidf权重
# 写入文本
with open("tfidf_word.txt", 'w', encoding='utf-8')as f:
    for i in range(len(weightlist)):
        f.write("第"+str(i)+"段文本："+"\n")
        # print("------第",i,'段文本词语的tfidf权重')
        for j in range(len(wordlist)):
            # print(wordlist[j],weightlist[i][j])
            f.write(wordlist[j]+"  "+str(weightlist[i][j])+"\n")

dict_final = {}
# count2=0
for i in range(len(weightlist)):
    for j in range(len(wordlist)):
        '''
        try:
            dict_final[wordlist[j]]+=weightlist[i][j]
        except:
            print(wordlist[j])
            # count2+=1
            dict_final[wordlist[j]]=0
            dict_final[wordlist[j]] += weightlist[i][j]
        '''
        #print('?')
        dict_final[wordlist[j]] = weightlist[i][j]

dict_final_sort = sorted(dict_final.items(), key=lambda x: x[1], reverse=True)
#print(dict(dict_final_sort))
with open("main_txt.txt",'w')as f:
    for i in list(dict_final_sort):
        every=str(i)
        # print(every)
        f.write(every+'\n')
# print(count2)
#pprint(dict_final_sort[:10])
#print(words)
