#Scritps_AP

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#Charger les données d'entraînement à partir du fichier CSV
train_data = pd.read_csv('dataframe_train.csv')

#Charger les données de test à partir du fichier CSV
test_data = pd.read_csv('dataframe_test.csv')

#Séparer les caractéristiques (X) et les étiquettes (y) pour l'entraînement
X_train = train_data['texte']  #Colonne contenant les données textuelles
y_train = train_data['parti']  #Colonne contenant les partis politiques

X_test = test_data['texte']    #Colonne contenant les données textuelles pour le test
y_test = test_data['parti']    #Colonne contenant les partis politiques pour le test

#Vectorisation des données textuelles si nécessaire (exemple avec CountVectorizer ou TF-IDFVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()  #On utilise TF-IDFVectorizer également
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Initialiser et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train_vec, y_train)

#Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test_vec)

#Évaluer les performances du modèle
accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle : {accuracy}")


#Calculer la précision
precision = precision_score(y_test, predictions, average='weighted')

#Calculer le rappel
recall = recall_score(y_test, predictions, average='weighted')

#Calculer la F-mesure
f_measure = f1_score(y_test, predictions, average='weighted')

#Afficher les résultats
print(f"Précision : {precision}")
print(f"Rappel : {recall}")
print(f"F-mesure : {f_measure}")


#Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)

#Afficher la matrice de confusion avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Classe prédite')
plt.ylabel('Classe réelle')
plt.title('Matrice de confusion')
plt.show()



#Définir les mesures
labels = ['Précision', 'Rappel', 'F-mesure', 'Précision du modèle']
values = [precision, recall, f_measure, accuracy]

#On fait un graphique à barres
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'purple'])
plt.xlabel('Mesures')
plt.ylabel('Valeurs')
plt.title('Mesures de performance du modèle')
plt.ylim(0, 1)  #Limite sur l'axe y pour les mesures de précision, rappel, F-mesure
plt.show()

