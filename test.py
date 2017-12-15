# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:48:45 2017

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
import gym
from gym import wrappers
import random

from matplotlib import animation
from environmentV2 import *
import networkx as nx


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
optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)
merged_summary = tf.summary.merge_all()


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, r"E:\Project\python_Project\RL\MIMA\MIMA1\tmp\model.ckpt")

### Play till we are dead
myagent.reset()
term = False
predicted_q = []
#frames = []
import time
rewardrec = 0

#myagent.env.changepath(2)
#myagent.env.renderstruct(0)

observation = myagent.fetchstate()
while not term:
    #rgb_observation = env.render(mode = 'rgb_array')
 #   frames.append(rgb_observation)
    pred_q = sess.run(predictedreward, feed_dict={networkstate:[observation]})
    #predicted_q.append(pred_q)
    action = np.argmax(pred_q)
    #observation, _, term, _ = myagent._step(action)
    reward,observation,term = myagent._step(action)
    print(observation)

    if action/myagent.env.pathnum < 1:
        myagent.env.renderstruct(True)
    else:
        myagent.env.renderstruct(False)
    time.sleep(0.5)
    rewardrec += reward
    print(pred_q)
    print(action,reward,rewardrec)
    print(myagent.env.t)