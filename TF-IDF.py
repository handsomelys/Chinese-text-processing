from departing_and_reducing_stopwords import Departing_word as dwd
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn import feature_extraction

word_raw = dwd('stopwords.txt','verse.txt','out.txt')
word_raw.departing()
words = word_raw.get_words()
'''
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(words)

#print(vectorizer.get_feature_names())
#print(vectorizer.vocabulary_)
#print(count.toarray())

transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())
'''
#token_pattern default r"(?u)\b\w\w+\b" atleast 2 words
#max_df/min_df  filter above max_df or below min_df words in comparison examples
#max_feature:int    the max phrases in the word-table
tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_df=0.8,min_df=0.1)   
tfidf_model = tfidf_vec.fit(words)
tfidf_matrix = tfidf_model.fit_transform(words)

#print(tfidf_model.get_feature_names())
print(tfidf_model.vocabulary_)
#print(tfidf_matrix)
print(tfidf_matrix.toarray())   #get the vector of phrases

