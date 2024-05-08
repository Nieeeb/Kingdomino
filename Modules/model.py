from dataloading import *
from sklearn.ensemble import RandomForestClassifier
import os
from joblib import dump, load

# Funktioner der træner og gemmer en ny RandomForest som en joblib fil
def train_new_model():
    print("Valid model not found, training a new one")
    data = load_data()
    x_train, y_train, x_test, y_test = complete_split(data)
    model = RandomForestClassifier(random_state=42, n_jobs=-1, criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=7, n_estimators=80)
    model.fit(x_train, y_train)
    dump(model, r"Modules/Saved Data/randomforest.joblib")
    return model

# Funktioner der indlæser en trænet model
# Hvis filen ikke er til stede, trænes en ny model der gemmes
def load_trained_model():
    path = r"Modules/Saved Data/randomforest.joblib"
    try:
        if os.path.isfile(path):
            print("Model File Found")
            model = load(path)
            if type(model) != type(RandomForestClassifier()):
                print("Wrong Model Type")
                model = train_new_model()
            return model
        else:
            model = train_new_model()
            return model
    except:
        model = train_new_model()
        return model
    
def main():
    model = load_trained_model()
    print(model)

if __name__ == "__main__":
    main()