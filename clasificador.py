import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

class Clasificador:
    def __init__(self) -> None:
        self.dataset = None
        self.animal_class = {}

    def Logistic_Regression(self):

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    
        model = LogisticRegression()
        
        model.fit(x_train, y_train)
            
        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)


        # Visualizar los datos

        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)




    def K_Nearest_Neighbors(self):

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
        
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        k = 3

        model = KNeighborsClassifier(n_neighbors=k)
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)

        # Visualizar los datos

        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)


    def Support_Vector_Machines(self):

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']

        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = SVC(kernel='sigmoid', C=1.0)
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)

        # Visualizar los datos

        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)

    def Naive_Bayes(self):

        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']
       
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = MultinomialNB()
        
        model.fit(x_train, y_train)
        
        
        y_pred = model.predict(x_test)


        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)

        # Visualizar los datos

        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)

    def redNeuronal(self):


        x = self.dataset.drop(['Animal', 'Class'], axis=1)
        y = self.dataset['Class']


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)
        
        # Visualizar los datos

        print("Exactitud del modelo: {:.2f}".format(accuracy))
        print("Informe de clasificación:")
        print(report)
        print(y_pred)
        

    def read_Dataset(self):
        
        self.dataset = pd.read_csv("sets/zoo.data", header=None, names=[
            'Animal', 'Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
            'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic', 'Catsize', 'Class'
        ])
            