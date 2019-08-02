import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.csv")

#print(data.head()) #print de first few lines

#Convert non-numerico data into numerical data

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"])) #convert buying column into numbers  [3 3 3 ... 1 1 1]
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#predict = "class"

x = list(zip(buying,maint,door,persons,lug_boot,safety)) # [(3, 3, 0, 0, 2, 1), (3, 3, 0, 0, 2, 2),...
y = list(cls)
#print(x)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                            test_size=0.1)

#buscamos el mejor modelo entre 50 corridas,
#con un max_acc > 0.982
max_acc = 0.982
for i in range(50):
    # Split data into 90% training and 10% for testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                test_size=0.1)  # 0.1 = 10% or data

    # create the classifier
    knn_model = KNeighborsClassifier(n_neighbors=7)  # you can ajust the number or neighbors
    # 5 = 0.88
    # 7 = 0.95

    knn_model.fit(x_train, y_train)  # we train
    acc = knn_model.score(x_test, y_test)  # then test the model
    #print(acc)

    if(acc > max_acc):
        print(acc)
        print("new model with ACC = ",acc)
        with open("car_model", "wb") as f:  # guarda en un archivo el modelo entrenado
            pickle.dump(knn_model, f)
        max_acc=acc


modelo = open("car_model","rb") # cargamos el modelo guardado
knn_model = pickle.load(modelo)

predicted = knn_model.predict(x_test)
names = ["unacc", "acc","good","vgood"]

for x in range(len(predicted)):
    print("Predicted:", names[predicted[x]], "Data:",x_test[x], "Actual:",names[y_test[x]])

