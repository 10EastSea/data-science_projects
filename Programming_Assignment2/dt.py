import sys
import operator
import math
import pandas as pd
from itertools import combinations

####################################################################################################

# Node 클래스
class Node:
  def __init__(self, attribute=None, class_label_value=None):
    # attribute: internal node -> 자식으로 내려갈때 비교할 attribute를 나타냄
    # class_label_value: leaf node -> leaf node인 경우, 현재 아이템이 무슨 class인지 나타냄
    self.attribute = attribute
    self.class_label_value = class_label_value

    # child_node: internal node인 경우, 자식노드들을 저장
    self.child_node = {}

  def is_leaf(self): # leaf 노드인지 확인
    if len(self.child_node) == 0: return True
    else: return False

  def get_class_label_value(self, sample): # leaf 노드로 가서 class_label_value를 반환
    # leaf node인 경우
    if self.is_leaf(): return self.class_label_value

    # leaf node가 아닌 경우
    attribute_value = sample[self.attribute]
    for child, node in self.child_node.items(): # child node 순회하면서 이동할 node 찾음 (child: 튜플형태로 attribute value가 들어가 있음)
      if attribute_value in child:
        return node.get_class_label_value(sample)

# Decision Tree 클래스
class DecisionTree:
  def __init__(self, train_dataset, attribute_selection_measure):
    # attribute_selection_measure: attribute selection measure가 무엇인지 저장
    # 종류: information_gain, gain_ratio, gini_index
    self.attribute_selection_measure = attribute_selection_measure

    # attribute_values: attribute가 가진 value들의 종류를, column별로 array 형태로 저장 
    # class_label: class label에 해당하는 attribute name 저장
    self.attribute_values = {}
    for attribute in train_dataset.columns[:-1]:
      self.attribute_values[attribute] = train_dataset[attribute].unique()
    self.class_label = train_dataset.columns[-1]

    self.root_node = None # root node 초기화
    self.model_construct(train_dataset) # model contruction: decision tree 만들기

  ##################################################

  ''' infomation gain '''
  def info(self, dataset):
    info_D = 0.0
    for class_label_value, C_i in dataset[self.class_label].value_counts().iteritems():
      p_i = C_i / len(dataset)
      info_D += (-1.0) * p_i * math.log2(p_i + 1e-9) # log2(0) 오류 막기 위해, trivial 값 add
    return info_D

  def information_gain(self, attribute, dataset):
    gain_A, info_A_D = 0.0, 0.0
    for attribute_value in self.attribute_values[attribute]:
      new_dataset = dataset[dataset[attribute] == attribute_value]
      info_A_D += (len(new_dataset) / len(dataset)) * self.info(new_dataset)
    gain_A = self.info(dataset) - info_A_D
    return gain_A

  ''' gain ratio '''
  def split_info(self, attribute, dataset):
    splitinfo_A_D = 0.0
    for attribute_value in self.attribute_values[attribute]:
      new_dataset = dataset[dataset[attribute] == attribute_value]
      p_i = len(new_dataset) / len(dataset)
      splitinfo_A_D += (-1.0) * p_i * math.log2(p_i + 1e-9) # log2(0) 오류 막기 위해, trivial 값 add
    return splitinfo_A_D

  def gain_ratio(self, attribute, dataset):
    gain_A = self.information_gain(attribute, dataset)
    splitinfo_A_D = self.split_info(attribute, dataset)
    return gain_A / splitinfo_A_D

  ''' gini index '''
  def gini(self, dataset):
    gini_D = 1.0
    for class_label_value, C_i in dataset[self.class_label].value_counts().iteritems():
      p_i = C_i / len(dataset)
      gini_D -= p_i ** 2
    return gini_D

  def gini_index(self, attribute, dataset):
    # value_comb: value들을 binary partition으로 나눈 리스트
    attribute_value = self.attribute_values[attribute]
    value_comb = []
    for i in range(1, len(attribute_value)):
      if i*2 > len(attribute_value): break
      lefts = list(map(set, list(combinations(attribute_value, i))))
      for left in lefts:
        right = set(attribute_value) - left
        value_comb.append(tuple([tuple(left), tuple(right)]))
    # print(value_comb)

    # binary partition 선택
    asm_dict = {}
    gini_D = self.gini(dataset)
    for comb in value_comb:
      d1 = dataset[dataset[attribute].isin(comb[0])]
      d2 = dataset[dataset[attribute].isin(comb[1])]

      d1_D = len(d1) / len(dataset)
      d2_D = len(d2) / len(dataset)

      d1_gini = d1_D * self.gini(d1)
      d2_gini = d2_D * self.gini(d2)
      
      asm_dict[comb] = gini_D - (d1_gini + d2_gini) # gini index 적용
    asm_dict = sorted(asm_dict.items(), key=operator.itemgetter(1), reverse=True) # measure 값이 큰 것부터 내림차순
    # print(asm_dict[0])

    return asm_dict[0]

  ##################################################

  def get_majority(self, dataset):
    return dataset[self.class_label].value_counts().index[0] # value_counts()가 내림차순으로 정렬되므로 첫번째 index 값이 majority

  def select_attribute(self, dataset): # attribute 선택
    asm_dict = {}
    if self.attribute_selection_measure == "information_gain":
      for attribute in dataset.columns[:-1]:asm_dict[attribute] = self.information_gain(attribute, dataset)
      asm_dict = sorted(asm_dict.items(), key=operator.itemgetter(1), reverse=True) # measure 값이 큰 것부터 내림차순
    elif self.attribute_selection_measure == "gain_ratio":
      for attribute in dataset.columns[:-1]: asm_dict[attribute] = self.gain_ratio(attribute, dataset)
      asm_dict = sorted(asm_dict.items(), key=operator.itemgetter(1), reverse=True) # measure 값이 큰 것부터 내림차순
    elif self.attribute_selection_measure == "gini_index":
      for attribute in dataset.columns[:-1]: asm_dict[attribute] = self.gini_index(attribute, dataset)
      asm_dict = sorted(asm_dict.items(), key=lambda x: x[1][1], reverse=True) # measure 값이 큰 것부터 내림차순
    # print("asm_dict:", asm_dict)

    if self.attribute_selection_measure == "information_gain" or self.attribute_selection_measure == "gain_ratio": return asm_dict[0][0] # measure이 제일 큰 attribute name을 가져옴
    elif self.attribute_selection_measure == "gini_index": return [asm_dict[0][0], asm_dict[0][1]] # measure이 제일 큰 attribute name과 binary partion을 가져옴

  def create_node(self, dataset):
    # leaf node 처리
    if dataset[self.class_label].nunique() == 1: return Node(None, dataset[self.class_label].unique()[0]) # 남아있는 sample들이 전부 같은 클래스인 경우 (unique를 통해 확인)
    if len(dataset.columns) == 1: return Node(None, self.get_majority(dataset)) # 남아있는 attribute가 없는 경우 (class label column만 존재하는 경우) but, del 하지 않으므로 실행 안해도 됨

    # attribute 선택해서 노드 생성
    node = None
    selected_attribute = self.select_attribute(dataset)
    if self.attribute_selection_measure == "information_gain" or self.attribute_selection_measure == "gain_ratio": node = Node(selected_attribute, None)
    elif self.attribute_selection_measure == "gini_index": node = Node(selected_attribute[0], None)
    # print(selected_attribute)

    # child 생성
    if self.attribute_selection_measure == "information_gain" or self.attribute_selection_measure == "gain_ratio":
      for attribute_value in self.attribute_values[selected_attribute]:
        new_dataset = dataset[dataset[selected_attribute] == attribute_value] # 선택된 attribute value들로 dataset 필터링
        del new_dataset[selected_attribute]
        # print(new_dataset)
        if len(new_dataset) == 0: return Node(None, self.get_majority(dataset)) # new_dataset에 남아있는 sample이 없는 경우
        else: node.child_node[attribute_value] = self.create_node(new_dataset)
    elif self.attribute_selection_measure == "gini_index":
      # print(selected_attribute[1])
      for child_value in selected_attribute[1][0]:
        new_dataset = dataset[dataset[selected_attribute[0]].isin(list(child_value))] # 선택된 attribute value들로 dataset 필터링
        del new_dataset[selected_attribute[0]]
        # print(new_dataset)
        if len(new_dataset) == 0: return Node(None, self.get_majority(dataset)) # new_dataset에 남아있는 sample이 없는 경우
        else: node.child_node[child_value] = self.create_node(new_dataset)

    return node

  ##################################################

  def model_construct(self, train_dataset): # train dataset을 받으면, 그 샘플들에 대해 모델을 생성
    self.root_node = self.create_node(train_dataset)

  def model_usage(self, test_dataset): # test dataset을 받으면, 그 샘플들에 대해 모델을 통해 나온 class label을 생성
    class_label_list = []
    for i in range(len(test_dataset)):
      sample = test_dataset.loc[i]
      class_label_list.append(self.root_node.get_class_label_value(sample))
    test_dataset[self.class_label] = class_label_list
    return test_dataset

####################################################################################################

# 파일 읽기
train_filename = sys.argv[1] # dt_train.txt
test_filename = sys.argv[2] # dt_test.txt
output_filename = sys.argv[3] # dt_result.txt
train_dataset = pd.read_csv(train_filename, sep="\t")
test_dataset = pd.read_csv(test_filename, sep="\t")

# Decision Tree 실행
# 종류: information_gain, gain_ratio, gini_index
dt = DecisionTree(train_dataset, "gain_ratio")
result = dt.model_usage(test_dataset)
result.to_csv(output_filename, index=False, sep="\t")