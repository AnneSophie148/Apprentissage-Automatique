print('démarrage')

"A changer en fonction de si on veut avec où sans stopwords"

#path_train = '../output/dataframe_train.csv'
#path_test = '../output/dataframe_test.csv'
path_train = '../output/dataframe_train_without_stopwords.csv'

#ici on peut tester sur une langue specifique en ajoutant le code de la langue a la fin
path_test = '../output/dataframe_test_without_stopwords.csv'

from pandas import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd

def get_info(path):
    data = pd.read_csv(path, delimiter=',')
    parti = data['parti'].tolist()
    #attention sur les df without stopwords : changer content par texte
    texte = data['content'].tolist()
    doc = data['doc'].tolist()
    return parti, texte, doc


parti_train, texte_train, doc_train = get_info(path_train)
parti_test, texte_test, doc_test = get_info(path_test)



train_data = pd.DataFrame({'parti': parti_train, 'texte': texte_train})
test_data = pd.DataFrame({'parti': parti_test, 'texte': texte_test})

X_train, y_train = train_data['texte'], train_data['parti']
X_test, y_test = test_data['texte'], test_data['parti']


vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

#Entrainement du modele
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)

#Predictions
X_test_data = vectorizer.transform(test_data['texte'])
y_pred = nb_classifier.predict(X_test_data)

#Evaluation
accuracy = metrics.accuracy_score(test_data['parti'], y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

precision = precision_score(test_data['parti'], y_pred, average='weighted')
recall = recall_score(test_data['parti'], y_pred, average='weighted')
conf_matrix = confusion_matrix(test_data['parti'], y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
#Faire la matrice avec plt
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=test_data['parti'].unique())
disp.plot()
plt.show()

