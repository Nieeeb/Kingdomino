import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import ast
import seaborn as sns
IMAGE, TILE, LABEL, HSV = 0, 1, 2, 3

path = r"Modules/NimRod/hsv_training.csv"
df = pd.read_csv(path)
df['hsv'] = df['hsv'].apply(ast.literal_eval)

h = [i[0] for i in df['hsv']]
s = [i[1] for i in df['hsv']]
v = [i[2] for i in df['hsv']]

labels = np.unique(df['label'])
print(labels)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(data=df, xs=h, ys=s, zs=v, marker='o', c=pd.factorize(df['label'])[0])

#df.plot.scatter(x='hsv',y='hsv',z='hsv')

ax.set_xlabel('H')
ax.set_ylabel('S')
ax.set_zlabel('V')

plt.legend(title="Tile Type")

plt.show()
