import numpy as np
from pandas import *
import pandas as pd
import re


path_train_without_stopwords = '..\output\dataframe_train_without_stopwords.csv'
path_test_without_stopwords = '..\output\dataframe_test_without_stopwords.csv'



def get_info(path):
    # get information from the dataframe in lists
    data = pd.read_csv(path, delimiter=',')
    parti = data['parti'].tolist()
    texte = data['content'].tolist()
    doc = data['doc'].tolist()
    return parti, texte, doc

parti_train_stop, list_txt_train, doc_train_stop = get_info(path_train_without_stopwords)
parti_test_stop, list_txt_test, doc_test_stop = get_info(path_test_without_stopwords)


def tokenizer_and_normalizer(s):
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]


#stock of values in dic
dic_train = {}
for i in range(len(doc_train_stop)):
    dic_train[doc_train_stop[i]] = {}
    parti = parti_train_stop[i]
    if parti is not None and parti not in dic_train[doc_train_stop[i]]:
            dic_train[doc_train_stop[i]][parti] = [] 
    dic_train[doc_train_stop[i]][parti_train_stop[i]] += tokenizer_and_normalizer(list_txt_train[i])

dic_test = {}
for i in range(len(doc_test_stop)):
    dic_test[doc_test_stop[i]] = {}
    parti = parti_test_stop[i]
    if parti is not None and parti not in dic_test[doc_test_stop[i]]:
            dic_test[doc_test_stop[i]][parti] = [] 
    dic_test[doc_test_stop[i]][parti_test_stop[i]] += tokenizer_and_normalizer(list_txt_test[i])


from collections import Counter

def get_counts(doc):
    "compte du nombre de mots par document"
    for parti, content in doc.items():
        return Counter(content)
    
bows = [get_counts(dic_train[doc]) for doc in dic_train]

vocab = sorted(set().union(*bows))

w_to_i = {w: i for i, w in enumerate(vocab)}

bow_array = np.zeros((len(bows), len(vocab)), dtype=np.float16)

for i, bag in enumerate(bows):
    for w, c in bag.items():
        bow_array[i, w_to_i[w]] = c
print(bow_array)

import numpy as np

def get_class_probs(dico_train, liste_parti):
    "calcul la proba des classes"
    n_classes = len(liste_parti)
    counts = np.zeros(n_classes)
    for doc, value in dico_train.items():
        for parti, txt in value.items():
            indice_parti = liste_parti.index(parti)
            counts[indice_parti] += 1
    # Normalisation par la somme pour avoir les fréquences empiriques
    total = counts.sum()
    return counts/total

liste_parti = []
for doc, value in dic_train.items():
    for parti, txt in value.items():
        if parti not in liste_parti:
            liste_parti.append(parti)

P_classe = get_class_probs(dic_train, liste_parti)
print(P_classe)
print(liste_parti)

'''
#Repartition par classe

import matplotlib.pyplot as plt
plt.bar(liste_parti, P_classe)
plt.xlabel("Classe")
plt.ylabel("Proportion")
plt.title("Répartition des classes")
plt.show()
'''

bow_array.shape
vocabulary_size = bow_array.shape[1]

counts = np.zeros((len(liste_parti), vocabulary_size))

counts = np.ones(((len(liste_parti), vocabulary_size)))


for doc, value in dic_train.items():
    doc = doc.split(',')[0]
    if doc == 1:
        for parti, txt in value.items():
            indice_parti = liste_parti.index(parti)
            occurrences = bow_array[doc-1]
            counts[indice_parti] += occurrences



word_probs = np.empty((len(liste_parti), vocabulary_size))
for i, class_count in enumerate(counts):
    word_probs[i] = class_count/np.sum(class_count)
print(word_probs)

total_per_class = np.sum(counts, axis=1)




def get_word_probs(bows, dico_train, liste_parti):
    target = []
    for docu, value in dico_train.items():
        for parti, txt in value.items():
            t = liste_parti.index(parti)
            target.append(t)
    
    counts = np.ones((np.max(target)+1, bows.shape[1]))
    for doc, c in zip(bows, target):
        counts[c] += doc
    total_per_class = np.sum(counts, axis=1, keepdims=True)
    return counts/total_per_class



get_word_probs(bow_array, dic_train, liste_parti)

log_prior = np.log(get_class_probs(dic_train, liste_parti))
log_likelihood = np.log(get_word_probs(bow_array, dic_train, liste_parti))

n_classes = len(liste_parti)



def get_counts2(doc):
    "compte du nombre de mots par document 2"
    return Counter(doc)
    
def predict_class(doc):
    bow_dict = get_counts2(doc)
    bow = np.zeros(len(w_to_i))
    for w, c in bow_dict.items():
        if w not in w_to_i:
            continue
        bow[w_to_i[w]] = c
    class_likelihoods = np.matmul(log_likelihood, bow) + log_prior
    return np.argmax(class_likelihoods)

    

data_train = []
for doc, value in dic_train.items():
    for parti, txt in value.items():
        data_train.append(txt)


predictions_train = np.array([predict_class(text) for doc, value in dic_train.items() for text, parti in value.items()])
print("pred", predictions_train)

predictions_test = np.array([predict_class(text) for doc, value in dic_test.items() for text, parti in value.items()])
print("pred", predictions_test)



target = []
for docu, value in dic_train.items():
    for parti, txt in value.items():
        t = liste_parti.index(parti)
        target.append(t)


target_test = []
for docu, value in dic_test.items():
    for parti, txt in value.items():
        t = liste_parti.index(parti)
        target_test.append(t)

#Here global values True Positive (VP) or False Positice (FP)...

VP = 0
FP = 0
VN = 0
FN = 0

#the dictionnary stocks the value for each class and evaluate for each class the precision, recall and accuracy
dic_evaluate = {}
for party in liste_parti:
    dic_evaluate[party]={}
    dic_evaluate[party]['VP']=0
    dic_evaluate[party]['FP']=0
    dic_evaluate[party]['FN']=0
    dic_evaluate[party]['P']=0
    dic_evaluate[party]['R']=0
    dic_evaluate[party]['Fmesure']=0


#change predictions_train with predictions_test to predict on test data

for i, classe_predite in enumerate(predictions_train):
    V_classe = target[i]
    if V_classe == classe_predite:
        dic_evaluate[liste_parti[V_classe]]['VP']+=1
        VP += 1
    elif classe_predite != V_classe:
        FP+=1
        FN += 1
        dic_evaluate[liste_parti[classe_predite]]['FP']+=1
        dic_evaluate[liste_parti[classe_predite]]['FN']+=1

for party, measure in dic_evaluate.items():
    dic_evaluate[party]['P'] = dic_evaluate[party]['VP'] / (dic_evaluate[party]['VP'] + dic_evaluate[party]['FP']) if (dic_evaluate[party]['VP'] + dic_evaluate[party]['FP']) > 0 else 0
    dic_evaluate[party]['R'] = dic_evaluate[party]['VP'] / (dic_evaluate[party]['VP'] + dic_evaluate[party]['FN']) if (dic_evaluate[party]['VP'] + dic_evaluate[party]['FN']) > 0 else 0
    dic_evaluate[party]['Fmesure'] = 2 * (dic_evaluate[party]['P'] * dic_evaluate[party]['R']) / (dic_evaluate[party]['P'] + dic_evaluate[party]['R']) if (dic_evaluate[party]['P'] + dic_evaluate[party]['R']) > 0 else 0

macro_P = VP/(VP+FP)
macro_R = VP/(VP+FN) 
macro_F = 2*(macro_P*macro_R)/(macro_R+macro_P)
print(dic_evaluate)

print(macro_P, macro_R, macro_F)



#Evaluate with training data --> change for test to evaluate with test data

correct = predictions_train == target
print(f"Il y a {correct.sum()} exemples bien classés parmis {correct.size}, soit {correct.sum()/correct.size:.2%} d'exactitude")



