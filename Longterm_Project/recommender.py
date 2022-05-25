import sys
import numpy as np
import pandas as pd

class CollaborativeFiltering():
  def __init__(self, rating_matrix, threshold):
    self.rating_matrix = rating_matrix
    self.threshold = threshold
    self.num_of_users, self.num_of_items = self.rating_matrix.shape

    # user_id와 매트릭스의 인덱스, item_id와 매트릭스의 인덱스 매핑
    self.user_id_dict = {}
    for i, user_id in enumerate(rating_matrix.index): self.user_id_dict[user_id] = i
    self.item_id_dict = {}
    for i, item_id in enumerate(rating_matrix.columns): self.item_id_dict[item_id] = i

    print("Creating model...")
    self.user_rating_mean = np.zeros([self.num_of_users]) # 유저별 rating 평균
    for i in range(self.num_of_users): self.user_rating_mean[i] = np.mean(self.filtered_user(rating_matrix.iloc[i, :].values))

    self.correlation_table = np.zeros([self.num_of_users, self.num_of_users]) # 피어슨 유사도
    ''' ver 1 '''
    # np_rating_matrix = rating_matrix.values
    # for i in range(self.num_of_users):
    #   for j in range(self.num_of_users):
    #     self.correlation_table[i][j] = self.pearson_sim(np_rating_matrix[i], np_rating_matrix[j])
    #   if (i % 10) == 0: print(i,'/',self.num_of_users)
    ''' ver 2 '''
    self.correlation_table = np.corrcoef(rating_matrix) # 피어슨 유사도
    print("Model creation completed!")

  ################################################################################

  def filtered_user(self, user):
    return user[np.where(user > 0)]

  def cosine_sim(self, user1, user2): # 코사인 유사도
    numerator = np.dot(user1, user2)
    denominator = np.linalg.norm(user1) * np.linalg.norm(user2)
    return numerator / (denominator + 1e-9) # 분모 0 방지

  def pearson_sim(self, user1, user2): # 피어슨 유사도
    numerator = np.dot((user1 - np.mean(user1)), (user2 - np.mean(user2)))
    denominator = np.linalg.norm(user1 - np.mean(user1)) * np.linalg.norm(user2 - np.mean(user2))
    return numerator / (denominator + 1e-9) # 분모 0 방지

  ################################################################################
  
  def predict(self, test_dataset):
    result = []

    print("Predicting...")
    for sample in test_dataset.values.tolist():
      user_id = sample[0]
      user_idx = self.user_id_dict[user_id]
      predicted_rating = self.user_rating_mean[user_idx]
      
      item_id = sample[1]
      if item_id in self.rating_matrix.columns:
        neighbor_rating = self.rating_matrix[item_id].values
        
        mask1 = neighbor_rating > 0
        mask2 = self.correlation_table[user_idx] > self.threshold
        mask = mask1 * mask2
        # print(neighbor_rating[mask])
        # print(self.correlation_table[user_idx, mask])

        neighbor_rating = neighbor_rating[mask]
        neighbor_rating_mean = self.user_rating_mean[mask]
        neighbor_similarity = self.correlation_table[user_idx, mask]

        predicted_rating = neighbor_similarity * (neighbor_rating - neighbor_rating_mean)
        predicted_rating = self.user_rating_mean[user_idx] + (predicted_rating.sum() / (neighbor_similarity.sum() + 1e-9))
      
      if predicted_rating < 1: predicted_rating = 1
      elif predicted_rating > 5: predicted_rating = 5
      # else: predicted_rating = round(predicted_rating)
      result.append(predicted_rating)
    print("Prediction completed!")

    test_dataset = test_dataset.drop('time_stamp', axis=1)
    test_dataset['rating'] = result
    return test_dataset

# 파일 읽기
train_filename = sys.argv[1]
test_filename = sys.argv[2]
output_filename = train_filename + "_prediction.txt"

dataset_header = ['user_id', 'item_id', 'rating', 'time_stamp']
train_dataset = pd.read_csv(train_filename, sep="\t", names=dataset_header)
test_dataset = pd.read_csv(test_filename, sep="\t", names=dataset_header)

# rating matrix 만들기
tmp_train_dataset = train_dataset.drop('time_stamp', axis=1)
rating_matrix = tmp_train_dataset.groupby(['user_id', 'item_id'])['rating'].mean().unstack().fillna(0)

# Recommender System
cf = CollaborativeFiltering(rating_matrix, 0.03)
result = cf.predict(test_dataset)

# 파일 쓰기
result.to_csv(output_filename, header=False, index=False, sep="\t")