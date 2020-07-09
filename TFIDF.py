from departing_and_reducing_stopwords import Departing_word as dwd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn import feature_extraction
import numpy as np
from pprint import  pprint


np.set_printoptions(threshold=np.inf)

word_raw = dwd('stopwords.txt', 'verse.txt', 'out.txt')  # 测试文本
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
tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
tfidf_model = tfidf_vec.fit(corpus_text)
tfidf_matrix = tfidf_model.fit_transform(words_text)


print(tfidf_vec.get_feature_names())
# 词表
wordlist = tfidf_model.get_feature_names()
#print(len(wordlist))
# 权值表
weightlist = tfidf_matrix.toarray()
#print(len(weightlist))

# tf-idf矩阵中 元素a[i][j]表示j词在i类文本中的tfidf权重
# 写入文本
with open("tfidf_word.txt", 'w', encoding='utf-8')as f:
    for i in range(len(weightlist)):
        #f.write("第"+str(i)+"段文本："+"\n")
        # print("------第",i,'段文本词语的tfidf权重')
        for j in range(len(wordlist)):
            # print(wordlist[j],weightlist[i][j])
            f.write(wordlist[j]+"  "+str(weightlist[i][j])+"\n")

# 字典 键：词语 值：文本
# dict_tfidf = {}
# for i in range(len(weightlist)):
#     for j in range(len(wordlist)):
#         dict_tfidf[wordlist[j]] = weightlist[i][j]
# print(dict_tfidf)
# 排序字典
# dict_tfidf_sort = sorted(dict_tfidf.items(), key=lambda x: x[1], reverse=True)
# dict_tfidf_sort = dict(dict_tfidf_sort)
# print(len(dict_tfidf))
# print(len(dict_tfidf_sort))
# print(dict_tfidf_sort['编者按'])
# with open("main_txt.txt",'w')as f:
#     for i in list(dict_tfidf_sort.items()):
#         every=str(i)
#         # print(every)
#         # print(every)
#         f.write(every+'\n')
# print(dict_tfidf_sort['乔治'])
dict_final = {}
# count=0
for i in range(len(weightlist)):
    for j in range(len(wordlist)):
        try:
            if weightlist[i][j]>dict_final[wordlist[j]]:
                dict_final[wordlist[j]]=weightlist[i][j]
            # dict_final[wordlist[j]]+=weightlist[i][j]
        except:
            # count+=1
            dict_final[wordlist[j]]=0
dict_tfidf_sort = sorted(dict_final.items(), key=lambda x: x[1], reverse=True)
with open("main_txt.txt",'w')as f:
    for i in list(dict_tfidf_sort):
        every=str(i)
        # print(every)
        f.write(every+'\n')
# print(count)
#pprint(dict_tfidf_sort[:10])
dict_final_second = {}
# count2=0
for i in range(len(weightlist)):
    for j in range(len(wordlist)):
        try:
            dict_final_second[wordlist[j]]+=weightlist[i][j]
        except:
            # count2+=1
            dict_final_second[wordlist[j]]=0
dict_final_second_sort = sorted(dict_final_second.items(), key=lambda x: x[1], reverse=True)
# print(count2)
#pprint(dict_final_second_sort[:10])
#print(words)
