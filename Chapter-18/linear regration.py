import matplotlib.pyplot as plt
import numpy as np

# Given data points
X = [3, 4, 5, 6, 7, 8]
Y = [0, 7, 17, 26, 35, 45]

N = len(X)

sumx = sumy = sumx2 = sumxy = 0
for i in range(N):
    sumx += X[i]
    sumy += Y[i]
    sumx2 += X[i]**2
    sumxy += X[i]*Y[i]
meanx = sumx / N
meany = sumy / N

# Calculate linear regression coefficients
a1 = (N*sumxy - sumx*sumy) / (N*sumx2 - sumx**2)
a0 = meany - a1 * meanx

# Create the linear regression equation
equation = f"y = {a0:.3f} {'-' if a1 < 0 else '+'} {abs(a1):.3f}x"

# Print the equation
print("The straight line equation:")
print(equation)

# Plot the given data points
plt.scatter(X, Y, label='Given Data Points')

# Plot the linear regression line
x_values = np.linspace(min(X), max(X), 100)
y_values = a0 + a1 * x_values
plt.plot(x_values, y_values, label=f'Linear Regression Line: {equation}', color='red')

# Customize the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()