import numpy as np
import random
from math import e
#计算余弦相似度
# 适应值函数未编写完成
# 更新粒子的位置和速度未完成
def cosine_similarity(vec1,vec2):
    dist = float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist

def getRandom(size):
    m_max,m_min = 0
    while true:
        m_1 = random.uniform(1,size)
        m_2 = random.uniform(1,size)
        if m_1 != m_2:
            break
    if m_1 > m_2:
        m_max = m_1
        m_min = m_2
    else:
        m_max = m_2
        m_min = m_1
    return m_max,m_min
    
def fit_fun(particles):
    m = []   #随机取 ？m<=size
    size = particles.getSize()
    m_max,m_min = getRandom(size)
    for i in range(m_min,m_max):
        m.append(int(i))
    fit_value = 0
    for i in m:
        for j in size:
            dist = cosine_similarity(particles.getPatical_list()[i].getPos(),particles.getPatical_list()[j].getPos())
            fit_value = fit_value + dist
    return fit_value


class Particle:
    def __init__(self,x_max,max_vel,dim):
        super().__init__()
        self._pos = [random.uniform(-x_max,x_max) for i in range(dim)]
        self._vel = [random.uniform(-max_vel,max_vel) for i in range(dim)]
        self._best_pos = [0.0 for i in range(dim)]
        #self._fitness_value = fit_fun(self._pos)
    
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
    def __init__(self,dim,size,iter_num,x_max,max_vel,theta,gama,best_fitness_value=float('-Inf'),c1=2,c2=2,w=1):
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

    def getSize(self):
        return self.size

    def getPatical_list(self):
        return self.patical_list

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
        value = fit_fun(self)
        if value < part.getFitness_value():
            part.setFitness_value(value)
            for i in range(self.dim):
                part.setBest_pos(i,part.getPos()[i])
        if value <self.getBest_fitness_value():
            self.setBest_fitness_value(value)
            for i in range(self.dim):
                self.setBest_position(i,part.getPos()[i])
    
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

    def update(self):
        for i in range(self.iter_num):
            if i < self.gama * self.iter_num:
                for part in self.patical_list:
                    self.update_vel(part)
                    self.update_pos_global(part)
                self.fitness_val_list.append(self.getBest_fitness_value())
            else:
                for part in self.patical_list:
                    self.update_vel(part)
                    self.update_pos_part(part)
                self.fitness_val_list.append(self.getBest_fitness_value())
        return self.fitness_val_list, self.getBest_position()
    

    def s_function_global(self,part,i):
        return 1/(1+pow(e,-(part.getVel()[i])))

    def s_function_part(self,part,i):
        if part.getVel()[i] <= 0:
            return 1 - 2/(1+pow(e,-(part.getVel()[i])))
        else:
            return 2/(1+pow(e,-(part.getVel()[i]))) - 1
