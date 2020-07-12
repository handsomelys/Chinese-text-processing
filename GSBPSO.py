import numpy as np
import random
from math import e
#计算余弦相似度
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)   #计算范数
    return num / denom

def get_cosine_sim(A,B):
    num = float(dot(mat(A), mat(B).T))
    denum = linalg.norm(A) * linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn  # 余弦值为[-1,1],归一化为[0,1],值越大相似度越大
    sim = 1 - sim  # 将其转化为值越小距离越近
    return sim

def fit_fun(x):
    pass

class Particle:
    def __init__(self,x_max,max_vel,dim):
        super().__init__()
        self._pos = [random.uniform(-x_max,x_max) for i in range(dim)]
        self._vel = [random.uniform(-max_vel,max_vel) for i in range(dim)]
        self._best_pos = [0.0 for i in range(dim)]
        self._fitness_value = fit_fun(self._pos)
    
    def setPos(self,i,value):
        self._pos[i] = value
    
    def getPos(self):
        return self._pos

    def setBest_pos(self,i,value):
        self._best_pos[i] = value

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
    def __init__(self,dim,size,iter_num,x_max,max_vel,theta,gama,best_fitness_value=float('Inf'),c1=2,c2=2,w=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.dim = dim  #粒子的维度
        self.size = size    #粒子个数
        self.iter_num = iter_num    #迭代次数
        self.x_max = x_max  #粒子范围
        self.max_vel = max_vel  #粒子最大速度
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  #种群最优位置
        self.fitness_val_list = []  #每次迭代最优适应值
        self.theta = theta
        self.gama = gama

        #初始化种群
        self.patical_list = [Particle(self.x_max,self.max_vel,self.dim) for i in range(self.size)]

    def setBest_fitness_value(self,value):
        self.best_fitness_value = value
    
    def getBest_fitness_value(self):
        return self.best_fitness_value

    def setBest_position(self,i,value):
        self.best_position[i] = value
    
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
            vel_value = self.w * part.getVel()[i] + self.c1 * random.random() * (part.getBest_pos()[i] - part.getPos()[i])\
                + self.c2 * random.random() * (self.getBest_position()[i] - part.getPos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.setVel(i,vel_value)
            

    def update_pos_global(self,part):   #更新位置 侧重全局搜索能力
        for i in range(self,dim):
            if abs(part.getVel()) >= self.theta:
                if part.getVel() >= 0 and self.getBest_position()[i] == 0:
                    part.setPos(i,0)
                elif abs(part.getVel()) <= 0 and self.getBest_position()[i] == 1:
                    part.setPos(i,1)
            elif abs(part.getVel()) < self.theta:
                if random.random() < s_function_global(part,i):
                    part.setPos(i,1)
                else:
                    part.setPos(i,0)
    
    def update_pos_part(self,part):
        for i in range(self,dim):
            if part.getVel()[i] < 0:
                if random.random() <= s_function_part(part,i):
                    pos_value = 0
                else:
                    pos_value = part.getPos()[i]
            else:
                if random.random() <= s_function_part(part,i):
                    pos_value = 1
                else:
                    pos_value = part.getPos()[i]
            part.setPos(i,pos_value)

    def s_function_global(self,part,i):
        return 1/(1+pow(e,-(part.getVel()[i])))

    def s_function_part(self,part,i):
        if part.getVel()[i] <= 0:
            return 1 - 2/(1+pow(e,-(part.getVel()[i])))
        else:
            return 2/(1+pow(e,-(part.getVel()[i]))) - 1
