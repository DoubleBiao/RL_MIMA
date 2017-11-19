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

class environment:
    field_width = 5
    field_length = 5
    
    def __init__(self,routernum,routerlink,hackers):
        """ networkstruct: the router graph in matrix form. 
        hackers: the hacked deck info. format: [router index, increasing constant, bound]"""
        self.Network = self.getstruct(routernum,routerlink)
        self.t = 0
        self.HackRec = np.zeros((self.Network.shape[0],3))
        self.Hacklist = hackers[:,0]
        self.HackW = 0.1
        self.renewhack(hackers)
        self.routernum = routernum
        self.timerange = 40
        self.setend()
        self.reset()
        
        routernum = self.Network.shape[0]
        link = np.nonzero(self.Network)
        link = np.swapaxes(link,0,1)
        

    
        G = nx.Graph()
        G.add_nodes_from(nx.path_graph(routernum))
        G.add_edges_from(link)
        #G=nx.relabel_nodes(G,label)
        self.pos = nx.spring_layout(G)
        
    def setend(self):
        self.endrouter = random.choice(range(self.routernum))
        while(len(np.nonzero(self.Hacklist == self.endrouter)[0]) != 0):
            self.endrouter = random.choice(range(self.routernum))
    def reset(self):
        self.currentrouter = random.choice(range(self.routernum))
        while(self.endrouter == self.currentrouter):
            self.currentrouter = random.choice(range(self.routernum))
        self.startrouter = self.currentrouter
        
        return self.currentrouter,self.endrouter
        
    
    def getstruct(self,routernum,routerlink):
        netstruct = np.zeros([routernum,routernum],dtype = bool)
        netstruct[routerlink[:,0],routerlink[:,1]] = True
        netstruct[routerlink[:,1],routerlink[:,0]] = True
        netstruct[range(netstruct.shape[0]),range(netstruct.shape[0])] = True
        return netstruct
    
    def renewhack(self,hackers):
        """hackers: the hacked deck info. format: [increasing constant, bound, hacked time]"""
        for i in range(hackers.shape[0]):
            indx = np.int32(hackers[i,0])
            self.HackRec[indx,0:2] = hackers[i,1:]
            self.HackRec[indx,2] = self.t
        
    def step(self,action):
        done = False
        self.t += 1
        
        if self.currentrouter == action:
            reward = -2
            done = False
            return self.currentrouter,reward,done
        
        self.currentrouter = action
        reward = -1 - self.HackW * self.getposs(self.currentrouter)
        
        self.t %= self.timerange
        
        
        if self.currentrouter == self.endrouter:
            done = True
            reward += 10
            return self.currentrouter,reward,done
        
        return self.currentrouter,reward,done
        
    def getposs(self,router):
        hackinfo = self.HackRec[router]
        return (1 - math.exp(-hackinfo[0]*(self.t - hackinfo[2])))*hackinfo[1]

    def getactlist(self,currentrouter):
        connect = self.Network[currentrouter,:]
        return np.nonzero(connect)[0]


    def renderstruct(self):
        label = dict()
        hackedlist = np.nonzero(self.HackRec[:,1])[0].tolist()
        
        for ind in hackedlist:
            poss = "p = {:.1f}".format(self.getposs(ind))
            label.update({ind:poss})
        
        nx.draw_networkx_nodes(G, self.pos,node_color = 'b')
        nx.draw_networkx_nodes(G, self.pos,nodelist = hackedlist)
        nx.draw_networkx_nodes(G, self.pos,nodelist = [self.endrouter],node_color = 'g')
        nx.draw_networkx_labels(G, self.pos)
        nx.draw_networkx_edges(G, self.pos)
        for key in label.keys():
            plt.text(self.pos[key][0] + 0.1,self.pos[key][1],label[key])
        plt.axis('off')
        plt.show()
        
    def render(self,router,lastrouter):

               
        label = dict()
        hackedlist = np.nonzero(self.HackRec[:,1])[0].tolist()
        
        for ind in hackedlist:
            poss = "p = {:.1f}".format(self.getposs(ind))
            label.update({ind:poss})
        
        nx.draw_networkx_nodes(G, self.pos,node_color = 'b')
        nx.draw_networkx_nodes(G, self.pos,nodelist = hackedlist)
        nx.draw_networkx_nodes(G, self.pos,nodelist = [self.endrouter,self.startrouter],node_color = 'g')
        nx.draw_networkx_nodes(G, self.pos,nodelist = [router],node_color = 'y')
        nx.draw_networkx_nodes(G, self.pos,nodelist = [lastrouter],node_color = 'k')
        nx.draw_networkx_labels(G, self.pos)
        nx.draw_networkx_edges(G, self.pos)
        nx.draw_networkx_edges(G, self.pos,width=6,edgelist = [(router,lastrouter)],node_color = 'y')
        for key in label.keys():
            plt.text(self.pos[key][0] + 0.1,self.pos[key][1],label[key])
        plt.axis('off')
        plt.show()
       # print(' ')
        
class agent():
    def __init__(self,env):
        self.env = env
    
    def randomdec(self):
        return random.choice(self.env.getactlist(self.env.currentrouter))
        
    def move(self,policyfunc):
        episode_reward = 0
        
        state,_ = self.env.reset()
        
        done = False
        while not done:
            #action = np.argmax(Q[state])
            action = policyfunc()
            laststate = state
            state, reward, done = self.env.step(action)
            episode_reward += reward
            
            print(reward)
            self.env.render(state,laststate)
            time.sleep(0.5)  
            #print(episode_reward)

        print ("Episode reward: %f" %episode_reward)

    def randomwalk(self):
        self.move(self.randomdec)

    
    
class Qagent(agent):
    def __init__(self,env,possres):
        agent.__init__(self,env)
        self.possres = possres
        self.posgrid = np.int32(100/self.possres)
        self.statenum = np.int32(env.routernum * self.posgrid**self.env.routernum)
        self.statetable = np.ones([self.statenum,env.routernum],dtype = np.float64)
        actind = np.tile(self.env.Network,(self.posgrid**self.env.routernum,1))
        self.statetable[actind] = 0
    
    def statemap(self,routerind):
        strid = env.routernum
        return routerind + strid*self.possind()
        
    def possind(self):
        ind = 0
        for i in range(self.env.routernum):
            ind += self.posgrid ** i * np.int32(self.env.getposs(i)/self.possres)
        return ind
    
    def e_policy(self,epsilon,stateind):
        actlist = self.env.getactlist(stateind).tolist()
        stateQ = self.statetable[stateind,actlist]
        
        maxind = np.argmax(stateQ)
        maxact = actlist[maxind]
        actlist.pop(maxind)
        
        select = random.random()
        if select <= (1 - epsilon):
            return maxact
        else:
            return random.choice(actlist)
        
    def greedy(self):
        stateind = self.env.currentrouter
        actlist = self.env.getactlist(stateind).tolist()
        stateQ = self.statetable[stateind,actlist]

        maxind = np.argmax(stateQ)
        maxact = actlist[maxind]
        
        return maxact
        
    def train(self,num_episodes, gamma, lr, e):
        for episode in range(num_episodes):
            #episode_reward = 0
            router,_ = self.env.reset()
            done = False
            start_state = True
            while not done:
                action = self.e_policy(e,self.env.currentrouter)
                
                if start_state == False:
                    last_state = self.statemap(last_router)
                    state = self.statemap(router)
                    self.statetable[last_state,last_action] = \
                    self.statetable[last_state,last_action] + \
                    lr*(reward + gamma*np.max(self.statetable[state,:]) - \
                    self.statetable[last_state,last_action])
                    #Q[last_state][last_action] = Q[last_state][last_action] + lr*(reward + gamma*np.max(Q[state]) - Q[last_state][last_action])
                else:
                    start_state = False
                #episode_reward += reward
                last_router = router
                #last_state = state
                last_action = action
                
                router, reward, done = self.env.step(action)
                #step_roc += 1
                #reward_roc += reward
            #action = act_select(Q[state],e)
            
            action = self.e_policy(e,self.env.currentrouter)
            last_state = self.statemap(last_router)
            state = self.statemap(router)
            
            self.statetable[last_state,last_action] = \
            self.statetable[last_state,last_action] + \
            lr*(reward + gamma*np.max(self.statetable[state,:]) -\
            self.statetable[last_state,last_action])

    def Qtest(self):
        self.move(self.greedy)
            #mean_reward += (reward_roc - mean_reward)/(episode + 1)
            
            #QLearning_reward_list.append(mean_reward)
            #QLearning_step_list.append(step_roc)
        
    
     
#G = nx.connected_caveman_graph(4, 5)
z =[4,4,4,3,2,4,4,6,5,4,2,3,5]
#z =[4,4,4,4,4,4]
print(nx.is_graphical(z))
G = nx.random_degree_sequence_graph(z)

routerlink = np.array([e for e in G.edges()])
hackers = np.array([[0,0.1,80],[9,0.2,70],[2,0.3,30],[7,0.3,55],[5,0.3,60]])
#hackers = np.array([[0,0.1,80]])
env = environment(len(G.nodes()),routerlink,hackers)
env.renderstruct()
    
testagent = Qagent(env,50)

testagent.train(num_episodes=1000, gamma=0.95, lr=0.1, e=0.2)
testagent.Qtest()
#testagent.randomwalk()
