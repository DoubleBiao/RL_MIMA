# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:22:44 2017

@author: HuXiaotian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:38:40 2017

@author: HuXiaotian
"""

from time import gmtime, strftime
import threading
import time

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display
import tensorflow as tf
#import gym
#from gym import wrappers
import random

from matplotlib import animation
from environmentV2 import *
import networkx as nx
#from JSAnimation.IPython_display import display_animation

#env = gym.make('CartPole-v0')
#observation = env.reset()

edges = [ (0,1),(1,2),(2,3),(3,0),(0,2),
          (0,4),(4,5),(5,6),(6,2), 
          (4,7),(6,8),(7,8),
          (7,9),(9,10),(10,11),(11,12),(12,8) ]

G = nx.Graph()
G.add_nodes_from(range(13))
G.add_edges_from(edges)
routerlink = np.array([e for e in G.edges()])
hackers = np.array([0,3,5,7])#np.array([[0,0.1,80],[9,0.2,70],[2,0.3,30],[7,0.3,55],[5,0.3,60]])
env = environment(len(G.nodes()),routerlink,hackers,0.8,0.05,2,9,G)
env.renderstruct(1)
myagent = agent(env)
#myagent.randomwalk()


# Network input
networkstate = tf.placeholder(tf.float32, [None, 14], name="input")
networkaction = tf.placeholder(tf.int32, [None], name="actioninput")
networkreward = tf.placeholder(tf.float32,[None], name="groundtruth_reward")
action_onehot = tf.one_hot(networkaction, 8, name="actiononehot")

# The variable in our network: 
w1 = tf.Variable(tf.random_normal([14,135], stddev=0.35), name="W1")
w2 = tf.Variable(tf.random_normal([135,270], stddev=0.35), name="W2")
w3 = tf.Variable(tf.random_normal([270,108], stddev=0.35), name="W3")
w4 = tf.Variable(tf.random_normal([108,52], stddev=0.35), name="W4")
w5 = tf.Variable(tf.random_normal([52,8], stddev=0.35), name="W5")
#w6 = tf.Variable(tf.random_normal([52,8], stddev=0.35), name="W6")

b1 = tf.Variable(tf.zeros([135]), name="B1")
b2 = tf.Variable(tf.zeros([270]), name="B2")
b3 = tf.Variable(tf.zeros([108]), name="B3")
b4 = tf.Variable(tf.zeros([52]), name="B4")
b5 = tf.Variable(tf.zeros([8]), name="B5")
#b6 = tf.Variable(tf.zeros([8]), name="B6")

# The network layout
layer1 = tf.nn.relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,w2), b2), name="Result2")
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2,w3), b3), name="Result3")
layer4 = tf.nn.relu(tf.add(tf.matmul(layer3,w4), b4), name="Result4")
#layer5 = tf.nn.relu(tf.add(tf.matmul(layer4,w5), b5), name="Result4")
predictedreward = tf.add(tf.matmul(layer4,w5), b5, name="predictedReward")

# Learning 
qreward = tf.reduce_sum(tf.multiply(predictedreward, action_onehot), reduction_indices = 1)

loss = tf.reduce_mean(tf.square(networkreward - qreward))
tf.summary.scalar('loss', loss)
optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
merged_summary = tf.summary.merge_all()

#sess = tf.InteractiveSession()
#summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)
#sess.run(tf.global_variables_initializer())
#
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, r"E:\Project\python_Project\RL\MIMA\MIMA1\tmp\model.ckpt")
#optimizer = tf.train.RMSPropOptimizer(0.9).minimize(loss)
summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)



replay_memory = [] # (state, action, reward, terminalstate, state_t+1)
epsilon = 1.0
BATCH_SIZE = 128
GAMMA = 0.4
MAX_LEN_REPLAY_MEMORY = 60000
FRAMES_TO_PLAY = 60001
MIN_FRAMES_FOR_LEARNING = 1000
summary = None

myagent.reset()
firsttime = True

epsilon = 0.3


for i_epoch in range(FRAMES_TO_PLAY):
    
    ### Select an action and perform this
    ### EXERCISE: this is where your network should play and try to come as far as possible!
    ### You have to implement epsilon-annealing yourself
    
    #action = env.action_space.sample()
    
    
    #choose action
    if len(replay_memory) > MIN_FRAMES_FOR_LEARNING * 2: 
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:[observation]})  #tensorflow 
        pred_reward = pred_reward[0]
        select = random.random()
        if select <= (1 - epsilon):
            action = np.argmax(pred_reward)
        else:
            action = myagent.randomdec()
    else:
        action = myagent.randomdec()
    
        
        
    
    reward,newobservation,done = myagent._step(action)
#    myagent.computelrandtime(t,newobservation)
    
    
    if(firsttime == True):
        firsttime = False
    else:
        replay_memory.append((observation, action, reward, done, newobservation))
    
    
    #newobservation, reward, terminal, info = env.step(action)

    ### I prefer that my agent gets 0 reward if it dies
    if done: 
        firsttime = True
        myagent.reset()
        
    observation = newobservation
    
    
    
    ### Add the observation to our replay memory
    
    #replay_memory.append((observation, action, reward, terminal, newobservation))
    #myagent.history.insert(0,myagent.state)
    
    ### Reset the environment if the agent died
    #if done: 
        #newobservation = env.reset()
    #    myagent.reset()
    #observation = newobservation
    
    ### Learn once we have enough frames to start learning
    if len(replay_memory) > MIN_FRAMES_FOR_LEARNING: 
        #experiences = random.sample(replay_memory, BATCH_SIZE)
        experiences = random.sample(replay_memory, BATCH_SIZE)
        
        totrain = [] # (state, action, delayed_reward)
        
        ### Calculate the predicted reward
        nextstates = [var[4] for var in experiences]
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:nextstates})
        
        ### Set the "ground truth": the value our network has to predict:
        for index in range(BATCH_SIZE):
            state, action, reward, done, newstate = experiences[index]
            predicted_reward = max(pred_reward[index])
            
            #if terminalstate:
            #    delayedreward = reward
            #else:
            delayedreward = reward + GAMMA*predicted_reward   #yarget
            
            totrain.append((state, action, delayedreward))
            
        ### Feed the train batch to the algorithm 
        states = [var[0] for var in totrain]
        actions = [var[1] for var in totrain]
        rewards = [var[2] for var in totrain]
        
        #training network
        _, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={networkstate:states, networkaction: actions, networkreward: rewards})


        ### If our memory is too big: remove the first element
        if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                replay_memory = replay_memory[1:]

        ### Show the progress 
        if i_epoch%100==1:
            summary_writer.add_summary(summary, i_epoch)
        if i_epoch%1000==1:
            print("Epoch %d, loss: %f" % (i_epoch,l))
            print(pred_reward[:16])
            #print(states[:16])
            #print(len(replay_memory))    
saver = tf.train.Saver()
save_path = saver.save(sess, r"E:\Project\python_Project\RL\MIMA\MIMA1\tmp\model.ckpt")

 