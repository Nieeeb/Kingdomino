import cv2 as cv
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from Kingdomino.Modules.TileSplitter import get_tiles

#Usual path extraction
path = os.path.abspath(__file__+'/../../../') + f'\King Domino dataset\Full game areas\\4.jpg'
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

# X_train, X_test, y_train, y_test = train_test_split(hsv_values, labels, test_size=0.2, random_state=42)
# knn.fit(X_train, y_train)
#
# unknown_X_test = np.array(unknown_hsv_values)
# y_pred = knn.predict(X_test)
#
# print("Forudsagte labels:", y_pred)
# print("Sande labels:", y_test)
#
# #Evaluation metrikker
# # Beregn nøjagtighed
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Nøjagtighed: {accuracy}")
#
# # Beregn præcision
# precision = precision_score(y_test, y_pred, average='macro')
# print(f"Præcision: {precision}")
#
# # Beregn recall
# recall = recall_score(y_test, y_pred, average='macro')
# print(f"Recall: {recall}")
#
# # Beregn F1-score
# f1 = f1_score(y_test, y_pred, average='macro')
# print(f"F1-score: {f1}")
#
# print(classification_report(y_test, y_pred))

# Initialiser variabler til at gemme totalen af scores
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0

#number of iterations
noi = 100
for i in range(noi):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(hsv_values, labels, test_size=0.2, random_state=i)

    # Fit knn
    knn.fit(X_train, y_train)

    # Forudsige labels for testdata
    y_pred = knn.predict(X_test)

    # Beregn nøjagtighed
    accuracy = accuracy_score(y_test, y_pred)
    total_accuracy += accuracy

    # Beregn præcision
    precision = precision_score(y_test, y_pred, average='macro')
    total_precision += precision

    # Beregn recall
    recall = recall_score(y_test, y_pred, average='macro')
    total_recall += recall

    # Beregn F1-score
    f1 = f1_score(y_test, y_pred, average='macro')
    total_f1 += f1

joblib.dump(knn, 'knn_model.joblib')
print("Knn trained and saved successfully")

# Udskriv gennemsnit af scoresne
print(f"Gennemsnitlig nøjagtighed: {total_accuracy / noi}")
print(f"Gennemsnitlig præcision: {total_precision / noi}")
print(f"Gennemsnitlig recall: {total_recall / noi}")
print(f"Gennemsnitlig F1-score: {total_f1 / noi}")