# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:18:50 2019

@author: xinweiy
"""
# The code take centerline image(segmented worm),tip image and
# direction image to produce an outout centerline.

# Python program to find single source shortest paths 
# for Directed Acyclic Graphs Complexity :OV(V+E) 
from collections import defaultdict 
import numpy as np
import scipy 
import skimage.morphology
import matplotlib.pyplot as plt


# Graph is represented using adjacency list. Every 
# node of adjacency list contains vertex number of 
# the vertex to which edge connects. It also contains 
# weight of the edge 
class Graph: 
    def __init__(self,vertices): 
  
        self.V = vertices # No. of vertices 
  
        # dictionary containing adjacency List 
        self.graph = defaultdict(list) 


  
    # function to add an edge to graph 
    def addEdge(self,u,v,w): 
        self.graph[u].append((v,w)) 
  
  
    # A recursive function used by shortestPath 
    def topologicalSortUtil(self,v,visited,stack): 
  
        # Mark the current node as visited. 
        visited[v] = True
  
        # Recur for all the vertices adjacent to this vertex 
        if v in self.graph.keys(): 
            for node,weight in self.graph[v]: 
                if visited[node] == False: 
                    self.topologicalSortUtil(node,visited,stack) 
  
        # Push current vertex to stack which stores topological sort 
        stack.append(v) 
  
    def GetPathway(self, start, end, dist, parent):
        path = list()
        if dist[end]!=float("Inf"):
          while  end!=start :
            path.append(end)
            end = parent[end]
        else:
          print("end point not reached from start point")
        
        return path[::-1]
    
    
    ''' The function to find shortest paths from given vertex. 
        It uses recursive topologicalSortUtil() to get topological 
        sorting of given graph.'''
    def shortestPath(self, s, e): 
          
        # Mark all the vertices as not visited 
        visited = [False]*self.V 
        stack =[] 
  
        # Call the recursive helper function to store Topological 
        # Sort starting from source vertice 
        for i in range(self.V): 
            if visited[i] == False: 
                self.topologicalSortUtil(s,visited,stack) 
  
        # Initialize distances to all vertices as infinite and 
        # distance to source as 0 
        dist = [float("Inf")] * (self.V) 
        dist[s] = 0
        parent = [float("Inf")]* (self.V)
  
        # Process vertices in topological order 
        while stack: 
  
            # Get the next vertex from topological order 
            i = stack.pop() 
  
            # Update distances of all adjacent vertices 
            for node,weight in self.graph[i]: 
                if dist[node] > dist[i] + weight: 
                    dist[node] = dist[i] + weight
                    parent[node] = i
  
#        # Print the calculated shortest distances 
#        for i in range(self.V): 
#            print ("%d" %dist[i]) if dist[i] != float("Inf") else  "Inf" , 
#  
        paths = list()
        for i in range(len(e)):
          paths.append(self.GetPathway(s, e[i], dist, parent))
        return paths


class FindCenterline(object):
  def __init__(self, tip_r=5):
    self.tip_r = tip_r
    y,x = np.ogrid[-self.tip_r: self.tip_r+1, -self.tip_r: self.tip_r+1]
    self.mask = x**2+y**2 <= self.tip_r**2
    self.switcher = {0:[1,0],
                     1:[1,1],
                     2:[0,1],
                     3:[-1,1],
                     4:[-1,0],
                     5:[-1,-1],
                     6:[0,-1],
                     7:[1,-1]} 
    self.weight1 = 1
    self.weight2 = 10
  def GetCenterline(self, img_c, img_dir, img_tip):
    # Get vortex from the img_c
    
    cord_x, cord_y = np.where(img_c)
    # store the vertex index in the img_index image
    # pixel value <0 if not a worm, value is the index in vortex.
    self.img_index = np.copy(img_c) -1
    
    # Generate vortex
    self.Num_vort = len(cord_x)
    self.vort_cord = list()
    for i in range(self.Num_vort):
      self.vort_cord.append([cord_x[i],cord_y[i]])
      self.img_index[cord_x[i],cord_y[i]] = i
    self.vort_cord = np.array(self.vort_cord)
    # generate the graph 
    self.g = Graph(self.Num_vort)   
    
    # Get edges from the img_dir
    # go through all the vortex
    for i in range(self.Num_vort):
      dir_x,dir_y = self.vort_cord[i]
      direction = img_dir[:, dir_x, dir_y]
      self.AddEdge(i,direction)
      
    # Get the head and tail from img_tip
    head_pts = self.GetTip(img_tip[0,:,:].astype(np.float32))
    tail_pts = self.GetTip(img_tip[1,:,:].astype(np.float32))

    ends = list()
    for i in range(len(tail_pts)):
      ends.append(self.img_index[tail_pts[i][0],tail_pts[i][1]])
     #list of centerline from different head tips. 
    clines = list()
    # run the shortest path algorithm
    for i in range(len(head_pts)):
      cline_tails = list()
      start = self.img_index[head_pts[i][0],head_pts[i][1]]
      cline = self.g.shortestPath(start,ends)
      for j in range(len(cline)):
        self.vort_cord[cline[j]]
        cline_tails.append(self.vort_cord[cline[j]])
      clines.append(cline_tails)
    # output the centerline.
    return clines      
  
  def GetTip(self,img_tip):
    img_tip1 = scipy.signal.convolve2d(img_tip,self.mask)
    threshold, upper, lower = 10, 1, 0
    img_tip2 = np.where(img_tip1 >threshold, upper, lower)

    img_tip3 = skimage.morphology.remove_small_objects(img_tip2,30)
    label_tip ,num_label= skimage.morphology.label(img_tip3, return_num=True)
    tips = list()
    for i in range(num_label):
      x,y =np.where(label_tip==i+1)
      tips.append((int(np.mean(x)), int(np.mean(y))))
    return tips
  
  
  def AddEdge(self, i, direction):
    # add the ediges for the  vortex 
    if np.sum(direction):
      if np.sum(direction)==1:
         # only one direction is identified, then use the cone (including three direction)
        direction = np.where(direction)[0][0]
        displacement = self.switcher[direction]
        tmp_cord = np.array(self.vort_cord[i]) + np.array(displacement)
        label = self.img_index[tmp_cord[0],tmp_cord[1]]
        if label>0:
          self.g.addEdge(i, label, self.weight1) 
        # adjacent direction is also allowed.
        displacement = self.switcher[np.mod(direction-1,8)]
        tmp_cord = np.array(self.vort_cord[i]) + np.array(displacement)
        label = self.img_index[tmp_cord[0],tmp_cord[1]]
        if label>0:
          self.g.addEdge(i, label, self.weight2) 
        
        displacement = self.switcher[np.mod(direction+1,8)]
        tmp_cord = np.array(self.vort_cord[i]) + np.array(displacement)
        label = self.img_index[tmp_cord[0],tmp_cord[1]]
        if label>0:
          self.g.addEdge(i, label, self.weight2) 
          
        # more than one direction is identified. Just use them
      else:
        directions = np.where(direction)[0]
        for j in directions:
          displacement = self.switcher[j]
          tmp_cord = np.array(self.vort_cord[i]) + np.array(displacement)
          label = self.img_index[tmp_cord[0],tmp_cord[1]]
          if label>0:
            self.g.addEdge(i, label, self.weight2) 
        

if __name__ == "__main__":
  path_cline = "C:\\Users\\xinweiy\\Desktop\\github\\Pytorch-UNet\\106_cline.npy"
  img_c = np.load(path_cline)    
  path_direction = "C:\\Users\\xinweiy\\Desktop\\github\\Pytorch-UNet\\106_direction.npy"
  img_dir = np.load(path_direction)    
  path_tip = "C:\\Users\\xinweiy\\Desktop\\github\\Pytorch-UNet\\106_tip.npy"
  img_tip = np.load(path_tip)  
  

  selem = skimage.morphology.disk(5)
  img_c_erosion = skimage.morphology.erosion(img_c[0,:,:], selem) + img_tip[0,:,:]
  #selem = skimage.morphology.disk(5)
  for i in range(len(img_dir)):
    img_dir[i,:,:] = skimage.morphology.binary_dilation(img_dir[i,:,:],selem)
    
    
  fCline = FindCenterline(tip_r=3) 
  clines = fCline.GetCenterline(img_c_erosion,img_dir,img_tip)
  
  num_head = len(clines)
  for i in range(num_head):
    clines_tail = clines[i]
    num_tail = len(clines_tail)
    for j in range(num_tail):
      cline = clines_tail[j]
      plt.imshow(img_c[0,:,:])
      plt.plot(cline[:,1],cline[:,0])
      plt.show()
    
    