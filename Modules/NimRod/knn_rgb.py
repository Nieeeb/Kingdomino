import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Læs csv'en med det labelt data
df = pd.read_csv('rgb_training.csv')

# extraction som np arrays
labels = np.array(df['label'].values)
rgb_values_strs = np.array(df['rgb'].values)
# tuplerne i datasættet er åbenbart string lmao
rgb_values = [ast.literal_eval(medians) for medians in rgb_values_strs]

# initier knn
knn = KNeighborsClassifier(n_neighbors=3)
# X_train, X_test, y_train, y_test = train_test_split(rgb_values, labels, test_size=0.2, random_state=42)
# knn.fit(X_train, y_train)
#
# # unknown_X_test = np.array(unknown_hsv_values)
# y_pred = knn.predict(X_test)
#
# print("Forudsagte labels:", y_pred)
# print("Sande labels:", y_test)
#
# print(classification_report(y_test, y_pred))
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
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0

#number of iterations
noi = 200
for i in range(noi):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(rgb_values, labels, test_size=0.2, random_state=i)

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

# Udskriv gennemsnit af scoresne
print(f"Gennemsnitlig nøjagtighed: {total_accuracy / noi}")
print(f"Gennemsnitlig præcision: {total_precision / noi}")
print(f"Gennemsnitlig recall: {total_recall / noi}")
print(f"Gennemsnitlig F1-score: {total_f1 / noi}")
