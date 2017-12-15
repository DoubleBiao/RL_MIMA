# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:35:24 2017

@author: HuXiaotian
"""
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import pathes



class environment:   
    def __init__(self,routernum,routerlink,hackers,accuracy,FOR,start,end,G):
        """ networkstruct: the router graph in matrix form. 
        hackers: the hacked deck info. format: [router index]"""
        self.Network = self.getstruct(routernum,routerlink)  #fetch the connectivity matrix
        
        self.t = 0.0                                           #overall time
        self.timerange = 1e5
        self.hackermoveT = 50.0
        #set hacked nodes, including the timeconstant, probility blablabla
        #need be changed into yesorno, accuracy, false omission rate 
        #self.HackRec = np.zeros((self.Network.shape[0],2))  
        self.Hacklist = hackers
        self.accuracy = accuracy
        self.FOR = FOR
        self.HackW = 0.1
#        self.renewhack(hackers)
        
        self.routernum = routernum
  
        self.G = G
        #pos should be pre-computed
        self.pos = nx.spring_layout(G)

        self.setcommunicatepair(start,end)
    def randomchangehackers(self):
        hackerlist = np.random.choice(self.routernum,5,replace = False)
        self.Hacklist = hackerlist
    def hackermove(self):
        tmpa,tmpb = divmod(self.t,self.hackermoveT)
        if tmpb == 0:
            self.randomchangehackers()
    
    
    def setcommunicatepair(self,start,end):
        self.start = start
        self.end = end
        self.pathtable = pathes.findoptionalpath(self.G,start,end)
        self.pathnum = len(self.pathtable)
        self.currentpath = np.random.choice(self.pathnum)
      
    
    def getstruct(self,routernum,routerlink):
        netstruct = np.zeros([routernum,routernum],dtype = bool)
        netstruct[routerlink[:,0],routerlink[:,1]] = True
        netstruct[routerlink[:,1],routerlink[:,0]] = True
        netstruct[range(netstruct.shape[0]),range(netstruct.shape[0])] = True
        return netstruct
 
    def changepath(self,pathind):
        self.currentpath = pathind
    
    def step(self):
        # rewrite the step function
        # return the positive or negtive of MIMA detection of nodes in the current
        # link nodes, and evaluate reward according to this detection
        self.t += 1
        def dodetection(nodes):
        # do MIMIA detection for nodes in the current path
        # according to the accuracy and FOR
            if nodes == self.start or nodes == self.end:
                return False
            
            if nodes in self.Hacklist:
            #    return True
#             if the nodes is hacked in reality, it will be detected in the probability 
#             equal to the accuracy
                if np.random.random() < self.accuracy:
                    return True
                else:
                    return False
            else:
#                return False
#             if the nodes is not hacked, it might be detected as hacked by mistake
#             in the probability equal to FOR(false emission error)
                if np.random.random() < self.FOR:
                    return True
                else:
                    return False
                
        routingnode = self.pathtable[self.currentpath]
        #currentpath = self.fetechpath()
        detecres = np.zeros(len(routingnode))
        
        for i in range(len(routingnode)):
            nodes = routingnode[i]
            if dodetection(nodes):
                detecres[i] = 1
        
        self.hackermove()
        return detecres,self.t
    
    def fetchoptionalpath(self):
        return self.pathtable
    def fetchdetectpara(self):
        return self.accuracy,self.FOR
    def fetchpath(self):
        return self.pathtable[self.currentpath]
    def fetchtime(self):
        return self.t
    def fetchpathnum(self):
        return len(self.pathtable)

#    def getactlist(self,currentrouter):
#        connect = self.Network[currentrouter,:]
#        return np.nonzero(connect)[0]


    def renderstruct(self,sendornot):
        #sendornot: send valid packages or not
        
        hackedlist = self.Hacklist.tolist()#np.nonzero(self.HackRec[:,1])[0].tolist()
        pathlist = self.pathtable[self.currentpath]
        routedges = [(pathlist[i],pathlist[i+1]) for i in range(len(pathlist) -1)]
        
        nx.draw_networkx_nodes(self.G, self.pos,node_color = 'b')
        
        if sendornot == True:
            nx.draw_networkx_nodes(self.G, self.pos,nodelist = pathlist,node_color = 'g')
            nx.draw_networkx_edges(self.G, self.pos,width=6,edgelist = routedges,edge_color  = 'g')
        else:
            nx.draw_networkx_nodes(self.G, self.pos,nodelist = pathlist,node_color = 'y')
            nx.draw_networkx_edges(self.G, self.pos,width=6,edgelist = routedges,edge_color  = 'y')
        nx.draw_networkx_nodes(self.G, self.pos,nodelist = [self.start,self.end],node_color = [.5,.5,0])      
        nx.draw_networkx_nodes(self.G, self.pos,nodelist = hackedlist)
        
      
        nx.draw_networkx_labels(self.G, self.pos)
        nx.draw_networkx_edges(self.G, self.pos)        
        plt.axis('off')
        plt.show()

        
class agent():
# in each step, store the detection history in the queues
# compute the likelihood from the history
# record the last renew time for each node
# evaluate the reward for each action
# 
    def __init__(self,env):
        self.env = env
        #frame length
        self.framelen =5
        #history memory
        self.history = []        
            #memory structure:
        #state memory:
        self.timespan = 10000.0
        
        self.successreward = 4
        self.testcost = -1
        self.punishment = -5
        
        self.state = np.zeros((3,env.routernum))
        
        acc,FOR = self.env.fetchdetectpara()
        self.lr1 = np.log(acc/FOR)
        self.lr0 = np.log((1 - acc)/(1 - FOR))

            #memory structure:
            #first row:   log likelihood ratio for all nodes
            #second row:  exp(last renew time - current time)
            #thrid  row:  last renew time ----> init with -99999
        
    def randomdec(self):
        actlist = list(range(2 * self.env.fetchpathnum()))
        return random.choice(actlist)
 
    
    def step(self,action):
        done = False
        
        (sendornot,pathid) = divmod(action,self.env.pathnum)
        #pathnum = self.env.fetchpathnum
        
        #if action != 0:
        self.env.changepath(pathid)
    
        #if change road: reset the current path of env
        detecres,t = self.env.step()
        #run on env step and get the detection return
        reward = self.rewardcompute(detecres,sendornot)
        
        if t == self.timespan:
            done = True
        
        return reward,detecres,self.env.currentpath,t,done
    
    def _step(self,action):
        reward,detecres,currentpath,t,done = self.step(action)
        self.computelrandtime(t,detecres)
        state = self.fetchstate()
        return reward,state,done
    
    def rewardcompute(self,detecres,action):
    #evaluate the reward
    #reward composition: 
    # test packages cost + 
    # if(valid packages)*(if(attacked)*punishment +  if(not attacked) * reward
        detecres = np.array(detecres)
        
        if action == 0:
            #send valid packages
            attackedlist = np.nonzero(detecres == 1)[0]
            if(attackedlist.size == 0):
                reward = self.successreward
            else:
                reward = self.punishment
        else:
            reward = self.testcost
        
        return reward
        
    def reset(self):
        self.env.t = 0
        self.state[2,:] = 0
    
    def move(self,policyfunc):
        #run for a fixed time span
        #renew overall time
        #self.env.t = 0
        self.reset()
        done = False
        while(not done):
        #enter in the loop
            #make a decision -> send valid packages or send test packages or change roads
            act = policyfunc()
            #run the step and get the detecres
            reward,detecres,currentpath,t,done = self.step(act)
#            if act == 0:
#                self.env.renderstruct(True)
#            else:
#                self.env.renderstruct(False)
                
            self.computelrandtime(t,detecres)
            #renew the history memory in queue fashion
                #enqueue new res from the head
            self.history.insert(0,self.state)
            print(reward)
#            if self.history == None:
#                self.history = self.state
#            else:
#                #self.history.insert(0,[currentpath,detecres.tolist()])
#                np.vstack([self.state,self.history])
                #np.append(history,detecres,axis = 0)
            
            
            #store the dectecres and evaluate the likelihood rate and last renew time
        # if time out, done:
            
    def computelrandtime(self,t,detecres):
    #compute the likelihood ratio and last renew time:
    #compute likelihood rate:
    #get least (framelen) history
        #historyspan = self.history[:self.timespan,:]
    #sweep the least history and determine the node to compute
        #historyspan.reverse()
        #trecord = t - self.timespan + 1   
        currentpath = self.env.fetchpath()
        for i in range(len(currentpath)):
            nodes = currentpath[i]
            lasttime = self.state[2,nodes]
            if (t - lasttime) != 1:
            #not consecutive renew:
                self.state[0,nodes] = 0
            #print(currentpath)
            #print(detecres)
            lrtmp = self.state[0,nodes]
            if detecres[i] == 1:
                lrtmp  += self.lr1
            else:
                lrtmp  += self.lr0
            
            if np.abs(lrtmp) > self.lr1 * self.framelen:
                lrtmp = self.lr1 * self.framelen * np.sign(lrtmp) 
            self.state[0,nodes] = lrtmp
            
            self.state[2,nodes] = t
            
#        self.state[1,:] =self.state[2,:] - t #np.exp(self.state[2,:] - t)
        
    def fetchstate(self):
        statevec = self.state[0,:]
        state = np.zeros(14)
        state[0:13] = statevec
        #state[13:26] = self.state[1,:]
        state[13] = self.env.currentpath
        #state[self.env.Hacklist] = 1
        #state[13] = self.env.currentpath
        return state#self.state[0,:]#np.append(statevec,self.state[1,:])
    
    def randomwalk(self):
        self.move(self.randomdec)

        
#   
#edges = [ (0,1),(1,2),(2,3),(3,0),(0,2),
#          (0,4),(4,5),(5,6),(6,2), 
#          (4,7),(6,8),(7,8),
#          (7,9),(9,10),(10,11),(11,12),(12,8) ]
#
#G = nx.Graph()
#G.add_nodes_from(range(13))
#G.add_edges_from(edges)
#routerlink = np.array([e for e in G.edges()])
#hackers = np.array([0,9,2,7,5])#np.array([[0,0.1,80],[9,0.2,70],[2,0.3,30],[7,0.3,55],[5,0.3,60]])
#env = environment(len(G.nodes()),routerlink,hackers,0.8,0.05,2,9,G)
#env.renderstruct(1)
#myagent = agent(env)
#myagent.randomwalk()

