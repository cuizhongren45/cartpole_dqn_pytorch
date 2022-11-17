#!/usr/bin/env python3

import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch 
# human图像化界面，rgb-array数组，训练更快
env = gym.make("CartPole-v1",render_mode="human")
BATCH_SIZE = 32
LR = 0.01                         
EPSILON = 0.9 #随机选取的概率，如果概率小于这个随机数，就采取greedy的行为
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = env.unwrapped
N_ACTIONS = env.action_space.n #小车动作空间
# print(N_ACTIONS)
N_STATES = env.observation_space.shape[0] #实验观测空间
print(N_STATES)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

# 网络结构初始化
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# DQN算法
class DQN(object):
    def __init__(self):
        #DQN是Q-Leaarning的一种方法，但是有两个神经网络，一个是eval_net一个是target_net
        #两个神经网络相同，参数不同，是不是把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net,self.target_net = Net(),Net()
        
        self.learn_step_counter = 0 #学习步数计数器
        self.memory_counter = 0 #记忆库中位值的计数器
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES * 2 + 2)) #初始化记忆库
        #记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr = LR)
        self.loss_func = nn.MSELoss()#优化器和损失函数
        
    
    #接收环境中的观测值，并采取动作
    def choose_action(self,x):
        #x-->(array([ 0.02041975, -0.02956359, -0.04300894, -0.01443677], dtype=float32), {})
        
        #x[0]就是观测值
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        # print(x)
        # x = torch.unsqueeze(torch.FloatTensor(x[0]),0)
        # print(x)
        if np.random.uniform() < EPSILON:
            #随机值得到的数有百分之九十的可能性<0.9,所以该if成立的几率是90%
            #90%的情况下采取actions_value高的作为最终动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value,1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE) # return the argmax index
        else:
            #其他10%采取随机选取动作
            action = np.random.randint(0,N_ACTIONS) #从动作中选一个动作
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        # print(action)
        return action    

    
    #记忆库，存储之前的记忆，学习之前的记忆库里的东西
    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s, [a, r], s_))
        # print(transition)
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if(self.memory_counter < 2000):
            print(self.memory_counter)
    
    def learn(self):
         # target net 参数更新,每隔TARGET_REPLACE_ITE更新一下
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
         #targetnet是时不时更新一下，evalnet是每一步都更新

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :] 
         #打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

          # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward() #误差反向传播
        self.optimizer.step()
    
    def save_model(self,e):
        # save_state = {'target_net':self.target_net.state_dict(),'eval_net':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':e}
        torch.save(self.eval_net,"cartpole_dqn.pt")
        print("save model of ",e)

dqn = DQN()

print('\nCollection experience...')
path = "cartpole_dqn.pt"

model = torch.load(path)

for i_episode in range(400):
    s = env.reset() #得到环境的反馈，现在的状态
    s = s[0]
    ep_r = 0
    save_flag = False
    time = 0
    while True:
        time  = time + 1
        env.render() #环境渲染，可以看到屏幕上的环境
        input = torch.tensor(s)
        # print(input)
        # print(s)
        # with torch.no_grad():
        # outputs = model(s)
        output = model(input)
        # print(output.detach.numpy())
        a_ = np.array(torch.argmax(output))
        # print(np.array(a_))
        # a = dqn.choose_action(s) #根据dqn来接受现在的状态，得到一个行为
        s_,r,done,info,_ = env.step(a_) #根据环境的行为，给出一个反馈

        
        if done or (time > 500):
            break
        
        s = s_ # 现在的状态赋值到下一个状态上去