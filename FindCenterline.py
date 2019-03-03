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
#import skimage.morphology
from skimage import morphology
#import matplotlib.pyplot as plt



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
          while  end!=start:
            path.append(end)
            end = parent[end]
        #else:
          #print("end point not reached from start point")

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
        step = [float("Inf")] * (self.V) 
        dist[s] = 0
        step[s] = 0

        parent = [float("Inf")]* (self.V)
  
        # Process vertices in topological order 
        while stack: 
  
            # Get the next vertex from topological order 
            i = stack.pop() 
  
            # Update distances of all adjacent vertices 
            for node,weight in self.graph[i]: 
                if dist[node] > dist[i] + weight: 
                    dist[node] = dist[i] + weight
                    step[node] = step[i] + 1

                    parent[node] = i
  
#        # Print the calculated shortest distances 
#        for i in range(self.V): 
#            print ("%d" %dist[i]) if dist[i] != float("Inf") else  "Inf" , 
#  
        paths = list()
        penalty = list()
        for i in range(len(e)):
          paths.append(self.GetPathway(s, e[i], dist, parent))
          penalty.append(dist[e[i]])
        return paths, penalty



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

    self.switcher2 = {0:[-1,-1],
                     1:[-1,0],
                     2:[-1,1],
                     3:[0,-1],
                     4:[0,0],
                     5:[0,1],
                     6:[1,-1],
                     7:[1,0],
                     8:[1,1]} 
    self.weight1 = 1
    self.weight2 = 10
    
    
  def skeletonize_cline(self, img_cline, head_last):
    sz = 5
    sz_dir = 10
    sz_open = 5

    #img_skel = morphology.skeletonize(img_cline)
    img_skel = morphology.thin(img_cline)

    (rows,cols) = np.nonzero(img_skel)
    # For each non-zero pixel...
    do_open = False
    for (r,c) in zip(rows,cols):
      # Extract an 8-connected neighbourhood
      (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
      # Convert into a single 1D array and check for non-zero locations
      pix_neighbourhood = img_skel[row_neigh,col_neigh].ravel() != 0
      # If the number of non-zero locations equals 2, add this to 
      # our list of co-ordinates
      if len(head_last):
        branch_head = head_last - np.array([[r,c]])
        dis_branch_head = np.sqrt(np.sum(branch_head**2))
      else:
        dis_branch_head = 0
        branch_head = np.array([[0,0]])
      # only deal with branch point that are close to head.          
      if np.sum(pix_neighbourhood) > 3 and dis_branch_head < 50:
        do_open = True
        if len(head_last):
          branch_head = branch_head / np.linalg.norm(branch_head, axis=1)[:, np.newaxis]      
        tmp = np.copy(img_skel[r-sz_dir:r+sz_dir+1, c-sz_dir:c+sz_dir+1])
        tmp[1:-1, 1:-1] = 0
        tmp = morphology.label(tmp)
        branch_pts = list()
        num_branches = np.max(tmp)
        for i in range(num_branches):
          (x,y) = np.nonzero(tmp==i+1) 
          branch_pts.append((x[0], y[0]))
        branch_pts = np.array(branch_pts) - sz_dir
        # normalize branch points direction
        branch_pts_norm = branch_pts/np.linalg.norm(branch_pts, axis=1)[:, np.newaxis]
        direct_branch = branch_pts_norm.dot(branch_pts_norm.T)
        # the score of strightness. higher score means not straight with others
        score_straight = np.sum(direct_branch, axis=0) - 1
        score_head = branch_pts_norm.dot(branch_head.T)[:,0]
        score = score_straight + (num_branches-1) * score_head
        #ind = np.unravel_index(np.argmin(direct_branch, axis=None), direct_branch.shape)
        ind = np.argsort(score)[0:2].tolist()
#        plt.figure(figsize=(20,10))
#        plt.imshow(img_cline)
#        print(head_last)
#        plt.scatter(head_last[1],head_last[0], s=50,c='black',alpha=0.5,marker='x')
#        plt.scatter(c,r, s=50,c='red',alpha=0.5,marker='x')
#        plt.show()
        for i in range(num_branches):
          if not i in ind:
            img_cline[r+branch_pts[i,0]-sz:r+branch_pts[i,0]+sz+1,
                      c+branch_pts[i,1]-sz:c+branch_pts[i,1]+sz+1] = 0
                    
#        print(direct_branch)
#        plt.figure(figsize=(20,10))
#        plt.imshow(tmp)
#        plt.show()
        

#    plt.figure(figsize=(20,10))
#    plt.imshow(img_cline)
#    plt.show()
#    
#    plt.figure(figsize=(20,10))
#    plt.imshow(img_skel)
#    plt.show()
#    
    if do_open:
        selem = morphology.disk(sz_open)
        img_cline = morphology.binary_opening(img_cline, selem)
#    plt.figure(figsize=(20,10))
#    plt.imshow(img_cline)
#    plt.show()

#    
    img_skel = morphology.skeletonize(img_cline)

#    plt.figure(figsize=(20,10))
#    plt.imshow(img_skel)
#    plt.show()
    
    return img_skel
    
    
    
  def cline_from_skel(self, img_c, img_dir, cline_dict, ref_head=[], ref_tail=[], head_last=[]):
    img_cline = np.copy(img_c[0,:,:])   
    self.img_dir = img_dir
    # Get the skeleton image from img_cline
    img_s = self.skeletonize_cline(img_cline, head_last)
    img_walk = np.copy(img_s)
    # get head and tail.   
    head_pts, tail_pts = self.skeleton_tip(img_s, ref_head=ref_head)    

    # build the graph with pixels on the centerline.
    cord_x, cord_y = np.where(img_s)
    # store the vertex index in the img_index image
    # pixel value <0 if not a worm, value is the index in vortex.
    self.img_index = np.copy(img_s) -1
    
    # Generate vortex
    self.Num_vort = len(cord_x)
    self.vort_cord = list()
    for i in range(self.Num_vort):
      self.vort_cord.append([cord_x[i],cord_y[i]])
      self.img_index[cord_x[i],cord_y[i]] = i
    self.vort_cord = np.array(self.vort_cord)
    
    ends = list()
    for i in range(len(tail_pts)):
      ends.append(self.img_index[tail_pts[i][0],tail_pts[i][1]])
      
    # generate the graph 
    self.g = Graph(self.Num_vort)   
   

    
    for i in range(len(head_pts)):
      start = self.img_index[head_pts[i,0], head_pts[i,1]]
      # Mark all the vertices as not visited 
      visited = [False]*self.Num_vort 
      # Call the recursive helper function to store Topological 
      # Sort starting from source vertice 
      self.add_edge_skel(start, visited, img_walk) 
      # Use the graph to run shortest path.  
      #cline_tails = list()
      
    # Use the graph to run shortest path.     
    #list of centerline from different head tips. 
    
    # run the shortest path algorithm
    for i in range(len(head_pts)):
      #cline_tails = list()
      start = self.img_index[head_pts[i][0],head_pts[i][1]]
      cline, penalty = self.g.shortestPath(start,ends)
     
      for j in range(len(cline)):
        #self.vort_cord[cline[j]]
        cline_dict["clines"].append(self.vort_cord[cline[j]])
        cline_dict["penalty"].append(penalty[j]/(len(cline[j])+1))
    # output the centerline.
    return cline_dict   


  def add_edge_skel(self, v, visited, img_walk):    
    # Mark the current node as visited. 
    visited[v] = True
    # label the pixel visted to be -1
    r = self.vort_cord[v,0]
    c = self.vort_cord[v,1]
    img_walk[r, c] = -1

    dir_score = np.where(self.img_dir[:, r, c])[0]
    directions = list()
    for i in range(len(dir_score)):
      directions.append(self.switcher[dir_score[i]])
      
    directions = np.array(directions)
    # Recur for all the vertices adjacent to this vertex 
    # Extract an 8-connected neighbourhood
    (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
    # Convert into a single 1D array and check for non-zero locations
    pix_neighbourhood = img_walk[row_neigh,col_neigh].ravel() > 0
    
    nodes_next = np.where(pix_neighbourhood)[0]
    
    for i in range(len(nodes_next)):
      step_next = np.array([self.switcher2[nodes_next[i]]])
      node = self.img_index[r+step_next[0,0], c+step_next[0,1]]
      # here not using any weight.
      if len(directions):
        if np.max(step_next.dot(directions.T))>0:
          self.g.addEdge(v, node, 1) 
        else:
          self.g.addEdge(v, node, 5) 
      else:
        self.g.addEdge(v, node, 3) 
      if visited[node] == False:
        self.add_edge_skel(node, visited, img_walk)
      
    
    
    
    
  def skeleton_tip(self, img_skel, ref_head=[]):
    # try get centerline from purely morphology method.
 
    (rows,cols) = np.nonzero(img_skel)
    # Initialize empty list of co-ordinates
    skel_coords = []
    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):
      # Extract an 8-connected neighbourhood
      (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
      # Convert into a single 1D array and check for non-zero locations
      pix_neighbourhood = img_skel[row_neigh,col_neigh].ravel() != 0

      # If the number of non-zero locations equals 2, add this to 
      # our list of co-ordinates
      if np.sum(pix_neighbourhood) == 2:
        skel_coords.append((r,c))
    skel_coords = np.array(skel_coords)    
    # identify heads and tails.
    # choose the one closest to ref_pt.
    if len(ref_head)==0:
      ref_head = [img_skel.shape[0]/2, img_skel.shape[1]/2] 
      #head_dis = 40
    #else:
     # head_dis = 40
    
    dist_head = np.sqrt( np.sum((skel_coords - ref_head)**2, axis=1) )
#    head_idx = [False]*len(skel_coords)
#    head_idx[np.argmin(dist_head)]= True
#
#    #head_pts = list()
#    head_pts = skel_coords[head_idx]
#    tail_pts = skel_coords[np.invert(head_idx)]
    
    head_idx = dist_head < 50
    head_pts = np.copy(skel_coords[head_idx])
    tail_pts = np.copy(skel_coords)


    
    return head_pts, tail_pts
    
 
  def GetCenterline(self, img_c, img_dir=[], img_tip=[]):
    # Get vortex from the img_c
    
    # try get centerline from skeleton morphology method.
    self.img_skel = morphology.skeletonize(img_c[0,:,:]) 
    # Get the head and tail from img_tip
    if len(img_tip) > 0:
      head_pts = self.GetTip(img_tip[0,:,:].astype(np.float32))
      tail_pts = self.GetTip(img_tip[1,:,:].astype(np.float32))
    else:
      self.skeleton_tip(self.img_skel)
      
      

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
    
    # Use the graph to run shortest path.

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

  def GetTip(self, img_tip, tip_ref=[]):
#    img_tip1 = scipy.signal.convolve2d(img_tip,self.mask)
#    threshold, upper, lower = 10, 1, 0
#    img_tip2 = np.where(img_tip1 > threshold, upper, lower)

    img_tip3 = morphology.remove_small_objects(img_tip>0, 100)
    label_tip ,num_label= morphology.label(img_tip3, return_num=True)
    tips = list()
    for i in range(num_label):
      x,y = np.where(label_tip==i+1)
      tips.append((int(np.mean(x)), int(np.mean(y))))
    
    tips = np.array(tips)
    if len(tip_ref)==0:
      tip_ref = [img_tip.shape[0]/2, img_tip.shape[1]/2 ]  
    if len(tips) > 1:
      dist = np.sum((tips - tip_ref)**2, axis=1)
      tips = tips[np.argsort(dist)]
      
    return np.array(tips)

  
  
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

          
          displacement = self.switcher[np.mod(j+1,8)]
          tmp_cord = np.array(self.vort_cord[i]) + np.array(displacement)
          label = self.img_index[tmp_cord[0],tmp_cord[1]]
          if label>0:
            self.g.addEdge(i, label, self.weight2)    
            
          displacement = self.switcher[np.mod(j-1,8)]
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
  

  selem = morphology.disk(5)
  img_c_erosion = morphology.erosion(img_c[0,:,:], selem) + img_tip[0,:,:]
  #selem = skimage.morphology.disk(5)
  for i in range(len(img_dir)):
    img_dir[i,:,:] = morphology.binary_dilation(img_dir[i,:,:],selem)

    
    
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
    
    