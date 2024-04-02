import cv2 as cv
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Andreas' tilesplitter funktion
def get_tiles(image):
    tiles = []
    for y in range(5):
        tiles.append([])
        for x in range(5):
            tile = image[y * 100:(y + 1) * 100, x * 100:(x + 1) * 100]
            tiles[-1].append(tile)
    return tiles

#Usual path extraction
path = os.path.abspath(__file__+'/../../../') + f'\King Domino dataset\Full game areas\\3.jpg'
unknown_image = cv.imread(path)
tiles = get_tiles(unknown_image)

#Brugt til at generere unlabeled HSV datapunkter til knn'en
unknown_hsv_values = []
for x, row in enumerate(tiles):
    for y, tile in enumerate(row):
        hsv_tile = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
        h, s, v = np.median(hsv_tile, axis=(0, 1))
        unknown_hsv_values.append([h, s, v])


#Læs csv'en med det labelt data
df = pd.read_csv('hsv_training.csv')

#extraction som np arrays
labels = np.array(df['label'].values)
hsv_values_strs = np.array(df['hsv'].values)
#tuplerne i datasættet er åbenbart string lmao
hsv_values = [ast.literal_eval(medians) for medians in hsv_values_strs]

#initier knn
knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(hsv_values, labels, test_size=0.2, random_state=42)
knn.fit(X_train, y_train)

unknown_X_test = np.array(unknown_hsv_values)
y_pred = knn.predict(unknown_X_test)

print("Forudsagte labels:", y_pred)
print("Sande labels:", y_test)

#Evaluation metrikker
# Beregn nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Nøjagtighed: {accuracy}")

# Beregn præcision
precision = precision_score(y_test, y_pred, average='macro')
print(f"Præcision: {precision}")

# Beregn recall
recall = recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall}")

# Beregn F1-score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-score: {f1}")
