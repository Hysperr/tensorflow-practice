"""
Python 3.6.2
"""
from math import log2
import matplotlib.pyplot as plt

values = [5, 4, 3, 2, 3, 4, 4]
prob_density = [x / 25 for x in values]
entropy = sum([x * log2(x) for x in prob_density]) * -1  # Entropy -Sigma(Pi * log2(Pi)), where Pi is pixel prob density

print("Intensity Range b: ", *range(1, 8))
print("Original Intensity Values N(b):", values)
print("Probability Density:", prob_density)
print("Image Entropy:", entropy)  # 2.7590795706241744

plt.hist(values, color="brown")
plt.title("Histogram of 5x5 Pixel Grid Intensities")
plt.xlabel("Intensity Value b")
plt.ylabel("Number of Pixels at Intensity Value N(b)")

plt.show()
