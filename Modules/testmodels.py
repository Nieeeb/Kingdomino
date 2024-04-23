from dataloading import *
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.semi_supervised import LabelPropagation
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import copy
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Funktion til at køre en grid search på en model og nogle givne parametre
# Opsat til at træne på træningssæt og teste på valideringssæt
# Fra Williams P1
def paramater_grid_search(model, params, x_train, y_train, x_validate, y_validate):
    # Laver kopier af alt data for at sikre der ikke bliver ændret på input
    x_train_working = copy.deepcopy(x_train)
    y_train_working = copy.deepcopy(y_train)
    x_validate_working = copy.deepcopy(x_validate)
    y_validate_working = copy.deepcopy(y_validate)
    
    # Samler trænings- og valideringsdata i et pandas dataframe til både x og y
    x = np.concatenate((x_train_working, x_validate_working))
    y = np.concatenate((y_train_working, y_validate_working))
    
    # Laver et dataframe fyldt med 0 med samme størrelse som træningssættet og et med 1 i samme størrelse som valideringssæt
    train_indices = np.zeros(len(x_train_working) - 1)
    validation_indixes = np.ones(len(x_validate_working))
    
    # Kombinerer de to dataframes. Hvis gridsearch ser 1 på en plads sættes den til at være testdata
    ps = PredefinedSplit(np.concatenate((train_indices, validation_indixes)))
    
    # Opretter gridsearchen. n_jobs=-1 gør at den bruger så meget computer den kan så det gør lidt hurtigere
    gs = GridSearchCV(model, params, cv=ps, return_train_score=True, scoring='f1_macro', n_jobs=-1)
    # Starter gridsearch
    gs.fit(x, y)
    
    # Få den bedste f1 score ud
    f1 = gs.best_score_
    # Printer de bedste parametre ud med den tilhørende score
    print(f"{model}: {gs.best_params_} F1: {f1}")
    return f1, gs.best_params_

def test_models(x_train, y_train, x_test, y_test):
    seed = 42
    
    knn = KNeighborsClassifier(n_jobs=-1)
    knn_params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    paramater_grid_search(knn, knn_params, x_train, y_train, x_test, y_test)
    
    svm = SVC(random_state=seed)
    svm_params = {
        'kernel': ['poly', 'rbf'],
        'degree': [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        'decision_function_shape': ['ovr', 'ovo']
    }
    paramater_grid_search(svm, svm_params, x_train, y_train, x_test, y_test)
    
    randomForest = RandomForestClassifier(random_state=seed, n_jobs=-1)
    forest_params = {
        'n_estimators': [75, 80, 85, 90],
        'criterion': ['entropy', 'gini'],
        'max_depth': [None, 7, 8, 9, 10],
        'min_samples_split': [5, 6, 7, 8],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', None]
    }
    paramater_grid_search(randomForest, forest_params, x_train, y_train, x_test, y_test)


def main():
    data = load_data()
    #x_train, y_train, x_validate, y_validate, x_test, y_test = complete_split(data, True)
    
    #test_models(x_train, y_train, x_validate, y_validate)
    
    x_train, y_train, x_test, y_test = complete_split(data)
    randomforest = RandomForestClassifier(random_state=42, n_jobs=-1, criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=7, n_estimators=80)
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_test)
    print(classification_report(y_pred=y_pred, y_true=y_test))

if __name__ == "__main__":
    main()
