import matplotlib.pyplot as plt
import numpy as np
import math
from manual_nn import run as predict

# Sample points in range 1-10 with decimals
x_values = np.linspace(1, 10, 100)
true_values = [math.log10(x) for x in x_values]
predicted_values = [predict(float(x))[0] for x in x_values]  # ensure float input

# Plot
plt.figure(figsize=(8,5))
plt.plot(x_values, true_values, label="True log10(x)", marker='o', markersize=3)
plt.plot(x_values, predicted_values, label="NN predicted", marker='x', markersize=3)
plt.xlabel("x")
plt.ylabel("log10(x)")
plt.title("Predicted vs True log10(x)")
plt.legend()
plt.grid(True)
plt.show()

