import re
line = "today  ,   is,     a ,  beautiful     , day"
words = re.split(r"\s*,\s*", line.strip())
wordSize = [len(w) for w in words]
