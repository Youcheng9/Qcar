import random
nameLst = ['Kevin', 'Ada', 'Jenny', 'Ade', 'Betty', 'Sam', 'Kenny', 'Beta', 'Berry', 'Terry', 'Nathan', 'Jess']
ageLst = [random.randint(18, 21) for _ in nameLst]
infoLst = list(zip(nameLst, ageLst))
print(infoLst)
print(sorted(infoLst))
print(sorted(infoLst, key=lambda p: (-p[1], p[0])))
# Test run 1: [('Kevin', 19), ('Ada', 20), ...]
# Test run 2: []
# Test run 3: [('Zed', 18), ...]
