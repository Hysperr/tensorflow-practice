import matplotlib.pyplot as plt
from numpy.random import normal

x = normal(size=200)
plt.hist(x, bins=30)
plt.show()

# Scratch work / Notes
print(", ".join(str(a) for a in range(10)))  # we are separating string by what we want, I choose ','
print(*range(10), sep='')  # Newer Python 3 way. We specify separator here to say no space. Leave it for normal space.

dictionary = {"nari": 5, "boli": False}
for i, k in dictionary.items():
    if i == "boli":
        print(k)

laundry = (1, 3, "string", 7.5)
print(laundry)
