fruitLst = ["melon", "banana", "banana", "melon", "apple", "melon", "cherry"]
Fruits = {}
for f in fruitLst:
    Fruits[f] = Fruits.get(f, 0) + 1
print(sorted(Fruits))
print(sum(Fruits.values()))
