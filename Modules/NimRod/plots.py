import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import ast
IMAGE, TILE, LABEL, HSV = 0, 1, 2, 3

df = pd.read_csv('hsv_training.csv')
df['hsv'] = df['hsv'].apply(ast.literal_eval)

h = [i[0] for i in df['hsv']]
s = [i[0] for i in df['hsv']]
v = [i[0] for i in df['hsv']]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(h, s, v, marker='o')

ax.set_xlabel('H')
ax.set_ylabel('S')
ax.set_zlabel('V')

vectors = df['hsv']
print(*vectors)

# Udpak koordinaterne i hver vektor
x, y, z = zip(*vectors)

# Opret en ny figur og 3D-akse
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')


# Plot vektorerne
ax1.quiver(0, 0, 0, x, y, z, length=255, color='blue', arrow_length_ratio=255)

# Sæt akseetiketter
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Sæt plot titel
plt.title('3D Vektor Visualisering')

plt.show()
