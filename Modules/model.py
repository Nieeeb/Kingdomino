import pickle
from dataloading import *
from sklearn.ensemble import RandomForestClassifier
import os

def pickle_dump(path, item):
    file = open(path, 'wb')
    pickle.dump(item, file)
    file.close()
    
# Funktion til at indlæse et pickle dump
def pickle_read_dump(path):
    file = open(path, 'rb')
    object = pickle.load(file)
    file.close()
    return object

def train_new_model():
    print("Valid model not found, training a new one")
    data = load_data()
    x_train, y_train, x_test, y_test = complete_split(data)
    model = RandomForestClassifier(random_state=42, n_jobs=-1, criterion='entropy', max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=7, n_estimators=80)
    model.fit(x_train, y_train)
    pickle_dump(r"Modules/Saved Data/randomforest", model)
    return model

def load_trained_model():
    path = r"Modules/Saved Data/randomforest"
    try:
        if os.path.isfile(path):
            print("File Found")
            model = pickle_read_dump(path)
            if type(model) != type(RandomForestClassifier):
                print("Wrong Type")
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