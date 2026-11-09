t = set()
t.update(["bye", "sad", "happy"])
lst = ["bye", "hi", "good", "okay", "sad"]
s = set(lst)
w = s & t
print(",".join(w))
