import sys
import math
import numpy as np
import pandas as pd

# Point 클래스
class Point:
  def __init__(self, object_id, x, y):
    self.object_id = int(object_id)
    self.x = float(x)
    self.y = float(y)
    self.cluster_id = None

  def dist(self, p):
    return math.sqrt((self.x-p.x)**2 + (self.y-p.y)**2)
  
  def to_print(self):
    print(self.object_id, self.x, self.y, self.cluster_id)


# DBSCAN 클래스
class DBSCAN:
  def __init__(self, dataset, eps, minPts):
    self.dataset = dataset
    self.eps = eps
    self.minPts = minPts

  def eps_neighborhood(self, p):
    return [q for q in self.dataset if p.dist(q) <= self.eps]
  
  def clustering(self):
    cluster_id = 0
    for p in self.dataset:
      if p.cluster_id is not None: continue # 이미 clustering 된 경우 
      
      neighbors = self.eps_neighborhood(p)
      if len(neighbors) >= self.minPts: # p가 core point인 경우
        p.cluster_id = cluster_id
        for q in neighbors:
          if q.cluster_id is None: # clustering 안된 경우
            q.cluster_id = cluster_id
            q_neighbors = self.eps_neighborhood(q)
            if len(q_neighbors) >= self.minPts: neighbors.extend(q_neighbors)
          elif q.cluster_id == -1: # q가 border point인 경우
            q.cluster_id = cluster_id # outlier 아님
        cluster_id += 1
      else: # p가 core point가 아닌 경우
        p.cluster_id = -1 # outlier 후보로 세팅
    
    clusters = [[] for i in range(cluster_id)]
    for p in self.dataset:
      # p.to_print()
      if p.cluster_id != -1: clusters[p.cluster_id].append(p)
      
    return clusters


# 파일 읽기
input_filename = sys.argv[1]
n = int(sys.argv[2])
eps = float(sys.argv[3])
minPts = float(sys.argv[4])

dataset_header = ["object_id", "x_coordinate", "y_coordinate"]
input_dataset = pd.read_csv(input_filename, sep="\t", names=dataset_header)

# 데이터 전처리
dataset = [Point(obj[0], obj[1], obj[2]) for obj in input_dataset.values.tolist()]
# for p in dataset: p.to_print() # 출력

# DBSCAN 클래스 생성 및 clustering
dbscan = DBSCAN(dataset, eps, minPts)
clusters = dbscan.clustering()

# 파일 쓰기
clusters.sort(key=len, reverse=True)
clusters = clusters[:n] # n개 만큼 선택

cluster_idx = 0
for cluster in clusters:
  with open(input_filename.split(".")[0] + "_cluster_" + str(cluster_idx) + ".txt", "w") as f:
    for p in cluster:
      f.write(str(p.object_id) + "\n")
  cluster_idx += 1