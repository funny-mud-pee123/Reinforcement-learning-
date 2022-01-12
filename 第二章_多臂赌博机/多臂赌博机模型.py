import numpy as np 
import matplotlib.pyplot as plt
import random

def Get_reward(state,action):#该函数用于从环境中获得奖励，输入为状态和行为,返回下一次状态和奖励
    if(action==0):
        reward = random.normalvariate( 2, 0.1)
    elif(action==1):
        reward = random.normalvariate( 1, 3)
    elif(action==2):
        reward = random.normalvariate( 2, 1)
    elif(action==3):
        reward = random.normalvariate( 0, 1)
    elif(action==4):
        reward = random.normalvariate( 3, 5)
    elif(action==5):
        reward = random.normalvariate( 5, 1)
    elif(action==6):
        reward = random.normalvariate( 7, 3)
    elif(action==7):
        reward = random.normalvariate( 4, 1)
    elif(action==8):
        reward = random.normalvariate( -1, 0.5)
    else:
        reward = random.normalvariate( 1, 1)
    state = state #只有一个状态0
    return [state,reward]


greedy = 0.8 #贪心概率
stop_number = 10000 #停止次数
state = 0 #初始状态

#建立一个Qtable存放十个行为价值的估计,使用乐观估计初始值100，N_A存放每一行为做了多少次
Q_table = 100*np.ones((1,10),np.float64) 
optimal_frequency = np.zeros((1,stop_number),np.float32)
N_A = np.zeros((1,10),np.float64) 


for i in range(stop_number):
    #选择行为 greedy-E policy---------------------------------------
    if (np.random.random() > greedy):
        action = np.random.randint(0,9)
    else:
        action = Q_table[state,:].argmax()
        

    #获得奖励，和环境交互--------------------------------------------
    [state,reward] = Get_reward(state,action)

    #更新次数，更新Qtable-------------------------------------------
    N_A[state,action] = N_A[state,action]+1
    Q_table[state,action] = Q_table[state,action]+(1/N_A[state,action])*(reward-Q_table[state,action])

    #记录选择最优动作的次数
    if action == 6:
        optimal_frequency[0,i:]=optimal_frequency[0,i:]+1
print(Q_table)
print(N_A)
print(optimal_frequency)

for i in range(stop_number):
    optimal_frequency[0,i] = optimal_frequency[0,i]/(i+1) #计算最优动作的概率

plt.scatter(np.arange(1,stop_number+1).reshape(1,stop_number),optimal_frequency) #绘图函数
plt.show()
