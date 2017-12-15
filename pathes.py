# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:26:48 2017

@author: HuXiaotian
"""

import networkx as nx



def findoptionalpath(G,src,dst):
    nodeToNodes = dict()
    mat = nx.adjacency_matrix(G)
    mat = mat.todense()
    for nodes in G.nodes():
        nodeToNodes[nodes] = []
        connectlist = mat[nodes].tolist()[0]
        for n in range(len(connectlist)):
            if connectlist[n] == 1:
               nodeToNodes[nodes].append(n)
               
    return getAllSimplePaths(src, dst, nodeToNodes,mat)
 
#
# Return all distinct simple paths from "originNode" to "targetNode".
# We are given the graph in the form of a adjacency list "nodeToNodes".
#
def getAllSimplePaths(originNode, targetNode, nodeToNodes,connectmat):
    return helpGetAllSimplePaths(targetNode,
                                 [originNode],
                                 set([originNode]),
                                 nodeToNodes,
                                 list(),connectmat)
 
#
# Return all distinct simple paths ending at "targetNode", continuing
# from "currentPath". "usedNodes" is useful so we can quickly skip
# nodes we have already added to "currentPath". When a new solution path
# is found, append it to "answerPaths" and return it.
#    
def helpGetAllSimplePaths(targetNode,
                          currentPath,
                          usedNodes,
                          nodeToNodes,
                          answerPaths,
                          connectmat):
    def findloop(currentpath,connectmat):
        if len(currentpath) < 3:
            return False
        
        thepath = currentpath.copy()
        currentnode = thepath[-1]
        thepath = thepath[:-2]
        thepath.reverse()
        
        for nodes in thepath:
            if connectmat[nodes,currentnode] == 1:
                return True
        
        return False
    
    lastNode = currentPath[-1]
    if lastNode == targetNode:
        answerPaths.append(list(currentPath))
    else:
        for neighbor in nodeToNodes[lastNode]:
            if neighbor not in usedNodes:
                currentPath.append(neighbor)
                usedNodes.add(neighbor)
                if findloop(currentPath,connectmat) == False:
                    helpGetAllSimplePaths(targetNode,
                                      currentPath,
                                      usedNodes,
                                      nodeToNodes,
                                      answerPaths,
                                      connectmat)
                usedNodes.remove(neighbor)
                currentPath.pop()
    return answerPaths


#findoptionalpath(G,5,9)