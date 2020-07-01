from departing_and_reducing_stopwords import Departing_word as dwd

word = dwd('stopwords.txt','verse.txt','out.txt')
word.departing()
words = word.get_words()
