import numpy as np
import random
from TFIDF import TextVectorizer
import math
from math import e
#计算余弦相似度
# 适应值函数未编写完成
# 更新粒子的位置和速度未完成
def cosine_similarity(x,y):

    num = float(np.matmul(x, y))
    s = np.linalg.norm(x) * np.linalg.norm(y)   #np.linalg.norm 默认是求整体矩阵元素平方和再开根号
    if s == 0:
       result = 0.0
    else:
          result = num/s
        
    return float(result)


def encode_of_textvector(textvector:list,chrom_a: list,chrom_b:list):
    '''
    textvector:关键词的文本向量，对应论文中的TF-IDF计算出的原始文本向量；
    chrom_a:代表染色体C1；
    chrom_b:代表染色体C2
    '''
    #print("="*60)
    #print(chrom_a)
    #print(chrom_b)
    union = [i | j for i, j in zip(chrom_a, chrom_b)] # 求得两条染色体的并集
    #print(union)
    dC1 = []
    dC2 = []
    for i in range(len(union)):
        if union[i]==1:
            if chrom_a[i]==1:
                dC1.append(textvector[i])
            else:
                dC1.append(0)
            if chrom_b[i] == 1:
                dC2.append(textvector[i])
            else:
                dC2.append(0)
    #print(dC1)
    #print(dC2)
    return dC1,dC2

def fit_fun(particles,x):
    '''
    适应值函数
    param：
        particles   粒子群
        x   随机小批量的个数
    '''
    fit_value_dict = {} 
    m = []   
    size = particles.getSize()
    #m_max,m_min = getRandom(size)
    if x == size:
        for i in range(x):
            m.append(int(i))
    else:
        for i in range(x):
            m.append(int(random.uniform(0,size)))   #获得需要更新适应值的个体
    
    for i in m:
        fit_value = 0
        for j in range(size):
            #fit_value = cosine_similarity(particles.getPatical_list()[i].getPos(),particles.getPatical_list()[j].getPos())
            dc1,dc2 = encode_of_textvector(feature_tfidf,particles.getPatical_list()[i].getPos(),particles.getPatical_list()[j].getPos())
            value = cosine_similarity(dc1,dc2)
            fit_value = fit_value + value
        if fit_value > particles.getPatical_list()[i].getFitness_value():   #获得更佳适应值
            particles.getPatical_list()[i].setBest_pos(particles.getPatical_list()[i].getPos()) #更新粒子的局部最优位置
        particles.getPatical_list()[i].setFitness_value(fit_value)  #为粒子更新当前的适应值
        fit_value_dict[i] = fit_value   #键：粒子编号   值：fitness_value
    fit_value_sort = sorted(fit_value_dict.items(), key=lambda x: x[1], reverse=True)   #排序 以便获得适应值最佳的粒子

    return fit_value_sort


class Particle:
    def __init__(self,max_vel,dim):
        super().__init__()
        self._pos = [np.random.randint(0,2) for i in range(dim)]    #初始化位置 随机0 1
        self._vel = [random.uniform(-max_vel,max_vel) for i in range(dim)]  #初始化速度
        self._best_pos = [0 for i in range(dim)]    #初始化最佳位置
        #self._fitness_value = fit_fun(self._pos)
        self._fitness_value = float('-inf')

    #   setters && getters
    def setPos(self,i,value):
        self._pos[i] = value
    
    def getPos(self):
        return self._pos

    def setBest_pos(self,value):
        self._best_pos = value

    def getBest_pos(self):
        return self._best_pos

    def setVel(self,i,value):
        self._vel[i] = value
    
    def getVel(self):
        return self._vel

    def setFitness_value(self, value):
        self._fitness_value = value

    def getFitness_value(self):
        return self._fitness_value
    
class GSBPSO:
    def __init__(self,dim,size,iter_num,max_vel,theta,gama,best_fitness_value=float('-Inf'),c1=2,c2=2,w=1):
        super().__init__()
        '''
        params:
            dim:粒子的维度
            size:粒子群的大小
            iter_num：迭代次数
            max_vel：速度范围
            theta:阈值
            gama:阈值
            c1:学习因子
            c2:学习因子
            w:学习惯性
        '''
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim  #粒子的维度
        self.size = size    #粒子个数
        self.iter_num = iter_num    #迭代次数
        #self.x_max = x_max  #粒子范围
        self.max_vel = max_vel  #粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  #种群最优位置
        self.fitness_val_list = []  #每次迭代最优适应值
        self.theta = theta
        self.gama = gama

        #初始化种群
        self.patical_list = [Particle(self.max_vel,self.dim) for i in range(self.size)]

        #初始化各个粒子的适应值
        '''
        init_fitvalue = fit_fun(self,self.size)
        self.setBest_fitness_value(init_fitvalue[0][1])
        self.setBest_position(self.getPatical_list()[init_fitvalue[0][0]])
        '''
    #   setters && getters
    def getSize(self):
        return self.size

    def getPatical_list(self):
        return self.patical_list

    def setBest_fitness_value(self,value):
        self.best_fitness_value = value
    
    def getBest_fitness_value(self):
        return self.best_fitness_value

    def setBest_position(self,value):
        self.best_position = value
    
    def getBest_position(self):
        return self.best_position

    def setTheta(self,value):
        self.theta = value

    def getTheta(self):
        return self.theta

    def setGama(self,value):
        self.gama = value

    def getGama(self):
        return self.gama

    def update_vel(self,part):
        for i in range(self.dim):
            #更新粒子速度
            vel_value = self.w*part.getVel()[i]+self.c1*random.random()*(part.getBest_pos()[i]-part.getPos()[i])+self.c2*random.random()*(self.getBest_position()[i]-part.getPos()[i])

            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.setVel(i,vel_value)
            

    def update_pos_global(self,part):   #更新位置 侧重全局搜索能力
        for i in range(self.dim):
            if abs(part.getVel()[i]) >= self.theta:
                if part.getVel()[i] >= 0 and self.getBest_position()[i] == 0:
                    part.setPos(i,0)
                elif abs(part.getVel()[i]) <= 0 and self.getBest_position()[i] == 1:
                    part.setPos(i,1)
            elif abs(part.getVel()[i]) < self.theta:
                if random.random() < self.s_function_global(part,i):
                    part.setPos(i,1)
                else:
                    part.setPos(i,0)

    def update_pos_part(self,part):
        for i in range(self.dim):
            if part.getVel()[i] < 0:
                if random.random() <= self.s_function_part(part,i):
                    pos_value = 0
                else:
                    pos_value = part.getPos()[i]
            elif part.getVel()[i] > 0:
                if random.random() <= self.s_function_part(part,i):
                    pos_value = 1
                else:
                    pos_value = part.getPos()[i]
            part.setPos(i,pos_value)

    def update(self):
        
        for i in range(self.iter_num):
            if i < self.gama * self.iter_num:
                for part in self.patical_list:
                    self.update_vel(part)
                    self.update_pos_global(part)
                #self.fitness_val_list.append(self.getBest_fitness_value())
            else:
                for part in self.patical_list:
                    self.update_vel(part)
                    self.update_pos_part(part)
                #self.fitness_val_list.append(self.getBest_fitness_value())
            update_fitvalue = fit_fun(self,5)
            if update_fitvalue[0][1] > self.best_fitness_value:
                self.best_fitness_value = update_fitvalue[0][1]
                #print(self.getPatical_list()[update_fitvalue[0][0]])
                self.best_position = self.getPatical_list()[update_fitvalue[0][0]].getPos()
                self.fitness_val_list.append(self.best_fitness_value)

        return self.fitness_val_list, self.getBest_position()
    

    def s_function_global(self,part,i):
        return 1/(1+pow(e,-(part.getVel()[i])))

    def s_function_part(self,part,i):
        if part.getVel()[i] <= 0:
            return 1 - 2/(1+pow(e,-(part.getVel()[i])))
        else:
            return 2/(1+pow(e,-(part.getVel()[i]))) - 1

if __name__ == "__main__":
    tv = TextVectorizer('corpus.txt','test_text.txt','stopwords.txt','corpus_output.txt','text_output.txt',max_df=0.15,min_df=0.0002)
    tv.init_corpus_and_text()
    tv.fitting()
    tv.transforming()
    wordlist = tv.getWordList()
    weightlist = tv.getWeightList()
    dict_ = tv.getDict()
    tv.writeInFile("file1.txt")
    tv.writeTfidfInFileSorted("file2.txt")
    textvector,feature_name,feature_tfidf = tv.getTextVector()

    dim = len(feature_name)
    #print(dim)
    size = dim * 5
    w = 1.4
    c1=c2=1.2
    gama = 0.9
    theta = 5
    max_vel = 1
    iter_num = 500
    gsbpso = GSBPSO(dim,size,iter_num,max_vel,theta,gama,c1=c1,c2=c2,w=w)

    fitness_value_list,best_position = gsbpso.update()
    print(best_position)
    print(fitness_value_list[-1])
    selected_feature = []
    for i in range(len(best_position)):
        if best_position[i] == 1:
            selected_feature.append(i)
    print(selected_feature)
        
    for i in selected_feature:
        print(feature_name[i])
    