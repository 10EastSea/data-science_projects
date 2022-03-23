import sys
from itertools import combinations

################################################################################

# 인자로 받은 number에 맞게, combination한 itemset을 반환
def comb(itemset, number):
  if number == 2:
    result = list(combinations(itemset, number)) # 모든 조합 구하기
    return list(map(list, result)) # list로 변환
  else:
    # itemset의 아이템들이 tuple로 되어 있음 -> tuple을 list로 바꿈 (2차원 리스트 형성)
    # 2차원 리스트를 1차원 리스트로 변경 (sum 이용)
    # 중복제거(set 이용) 이후, 다시 list로 변환
    all_itemset = list(set(sum(list(map(list, itemset)), [])))

    tmp_result = list(combinations(all_itemset, number)) # 모든 조합 구하기
    tmp_result = list(map(list, tmp_result)) # list로 변환

    # itemset에서 만들어질 수 없는 조합 제거
    result = []
    itemset = list(map(set, itemset)) # tuple로 묶여있던 아이템셋을 set으로 변환
    for candidate_item in tmp_result:
      items_comb = list(combinations(candidate_item, number-1)) # 만들어낸 candidate_item에서 만들 수 있는 조합 생성
      check = True
      for tmp_item in items_comb: # 그 조합들이 itemset에 있는지 확인
        tmp_item = set(tmp_item)
        if tmp_item not in itemset:
          check = False
          break
      if check: result.append(candidate_item) # 모든 조합이 itemset에 존재하는 경우

    return result

# candidate 아이템셋을 frequent 아이템셋으로 변경 (support 체크)
def candToFreq(candidate_itemset, minimum_support):
  frequent_itemset = []
  for item, sup in candidate_itemset.items():
    if sup >= minimum_support:
      frequent_itemset.append(item)
  return frequent_itemset

################################################################################


# 파일 읽기
transactions = [] # 파일을 읽은뒤, transaction들을 저장할 list

f = open(sys.argv[2], 'r')
lines = f.readlines()
for line in lines:
  transactions.append(list(map(int, line.split('\t'))))
f.close()
minimum_support = len(transactions) * (int(sys.argv[1]) * 0.01) # % 단위로 받은 minimum support를 갯수 단위로 환산하여 저장한 변수

# print(transactions)
# print('num of transactions:', len(transactions))
# print('minimum_support:', minimum_support)
# print('')


# 필요 변수 초기화 및 선언
k = 1           # 현재 아이템 조합 갯수 (k를 의미)
running = True  # 프로그램이 구동 중인지 확인하는 변수

candidate_itemset = {}  # candidate 아이템셋: dict로 관리
frequent_itemset = []   # frequent 아이템셋: array로 관리
result_itemset = []     # frequent 아이템셋 집합: 결과값


# 초기값(C1, L1 itemset) 만들기
for transaction in transactions: # C1 itemset 만들기
  for item in transaction:
    current_count = candidate_itemset.get(item, 0)
    candidate_itemset[item] = current_count + 1
frequent_itemset = candToFreq(candidate_itemset, minimum_support) # L1 itemset 만들기

# print('C 1 :', candidate_itemset)
# print('L 1 :', frequent_itemset)
# print('')


# apriori 알고리즘
if len(frequent_itemset) == 0: running = False
while running:
  k += 1 # 아이템 조합 갯수 하나 증가

  tmp_candidate_itemset = comb(frequent_itemset, k) # 가능한 모든 itemset 조합 (candidate_itemset 만들기 위한 전처리)
  if len(tmp_candidate_itemset) == 0:
    running = False
    break
  
  candidate_itemset = {} # 초기화
  for tmp_candidate_item in tmp_candidate_itemset: # Ck itemset 만들기
    current_count = 0
    for transaction in transactions:
      check = True
      for item in tmp_candidate_item: # transaction에서 tmp_candidate_item의 아이템들이 포함되는지 확인
        if item not in transaction:
          check = False
          break
      if check: current_count += 1 # tmp_candidate_item 여기에 있는 아이템들이 transaction에 있는 경우
    candidate_itemset[tuple(tmp_candidate_item)] = current_count
  frequent_itemset = candToFreq(candidate_itemset, minimum_support) # Lk itemset 만들기
  
  # print('C',k,':', candidate_itemset)
  # print('L',k,':', frequent_itemset)
  if len(frequent_itemset) == 0:
    running = False
    break

  result_itemset.extend(frequent_itemset)
  # print('result_itemset:', result_itemset)
  # print('')


# 파일 쓰기
f = open(sys.argv[3], 'w')
for itemset in result_itemset:
  itemset_list = list(itemset)
  for i in range(1, len(itemset_list)):
    # itemset_list에 있는 것들로 조합을 만듬 -> 만들어진 결과물: [튜플, 튜플, ..]
    # tuple을 set으로 바꿈
    lefts = list(map(set, list(combinations(itemset_list, i)))) # itemset에서 좌변에 해당할 집합

    for left in lefts:
      right = set(itemset) - left # itemset에서 차집합을 통해 우변에 해당할 집합을 생성

      # support, confidence 계산
      support = 0
      left_appear = 0
      right_appear = 0
      for transaction in transactions:
        transaction = set(transaction) # 부분집합으로 연산하기 위해 set으로 변경
        if set(itemset) <= transaction: support += 1 # transaction에 해당 itemset이 있으면 support 증가
        if left <= transaction: # transaction에 left가 존재하면서 이 때,
          left_appear += 1
          if right <= transaction: right_appear += 1 # right도 존재하면 confidence 증가
      support = '{:.2f}'.format(support/len(transactions) * 100)    # 포맷에 맞게 변경
      confidence = '{:.2f}'.format(right_appear/left_appear * 100)  # 포맷에 맞게 변경

      line = str(left)+'\t'+str(right)+'\t'+str(support)+'\t'+str(confidence)+'\n'
      f.write(line)
f.close()