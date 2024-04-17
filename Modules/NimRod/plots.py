import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

color_labels = df['label'].unique()
rgb_values = sns.color_palette("tab10", 8)
color_map = dict(zip(color_labels, rgb_values))

levels, categories = pd.factorize(df['label'])
colors = [plt.cm.tab10(i) for i in levels]

handles = [patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(h, s, v, c=df['label'].map(color_map))

ax.set_xlabel('H')
ax.set_ylabel('S')
ax.set_zlabel('V')

ax.legend(title='Tile Type', handles=handles)

plt.show()
