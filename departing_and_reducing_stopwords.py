import jieba 

#实现分词和去停用词
class Departing_word:
    stopwords = []
    words = []
    text_name = ''
    stopwords_file = ''
    output_file_name = ''
    def __init__(self,stopwords_file,text_name,output_file_name):
        super().__init__()
        self.stopwords_file = stopwords_file
        self.text_name = text_name
        self.output_file_name = output_file_name

    def stopwordslist(self):
        #获得停用词list
        self.stopwords = [line.strip() for line in open(self.stopwords_file,encoding='UTF-8').readlines()]
        #return stopwords

    def seg_depart(self,sentence):
        #分词
        sentense_depart = jieba.cut(sentence.strip())

        stopwords = self.stopwordslist()

        outstr = ''
        for word in sentense_depart:
            if word not in self.stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr
    def departing(self):

        inputs = open(self.text_name,'r',encoding='UTF-8')
        outputs = open(self.output_file_name,'w',encoding='UTF-8')

        words_raw = []

        for line in inputs:
            line_seg = self.seg_depart(line)
            outputs.write(line_seg + '\n')
            words_raw.append(line_seg)
            #将分词好的文本放到words_raw中
        outputs.close()
        inputs.close()
        
        words_raw1 = [i for i in words_raw if i != '']
        #去除空元素
        '''
        for i in words_raw1:
            self.words.extend(i.split())
            '''
        self.words = words_raw1
        #至此 words中存放分词后的文本
    def get_words(self):
        return self.words

