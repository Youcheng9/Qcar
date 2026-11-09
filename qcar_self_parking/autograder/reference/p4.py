inf = open("data.txt", "r")
try:
    for line in inf:
        x = float(line)
except ValueError:
    print("Wrong data format")
except:
    print("unexpected error")
else:
    print("Good")
finally:
    inf.close()
    print("Done")
