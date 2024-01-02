
from IPython.display import display, Markdown
import numpy as np
import xml.etree.ElementTree as ET
import csv

import os

def get_files(path):
    file_list = []
    for dirpath, dirs, files in os.walk(path):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith('.xml') or fname.endswith('.txt'):
                file_list.append(fname)
    return file_list

path_train = '..\deft09\Corpus_d_apprentissage'
path_test = '..\deft09\Corpus_de_test'
path_idee = '..\deft09\Donnée_de_référence'
train_files = get_files(path_train)
test_files = get_files(path_test)

import re
def tokenizer_and_normalizer(s):
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]

def parse_xml_train(file, dico):
    tree = ET.parse(file)
    root = tree.getroot()
    langue = file.split('_')[-1][:2]
    for i, doc in enumerate(root.findall('.//doc')):
        doc_id = (i+1, langue)
        dico[doc_id]= {}
        parti = doc.find('.//PARTI')
        texte = doc.find('.//texte')

        if parti is not None:
            valeur_parti = parti.get('valeur')
            if valeur_parti not in dico[doc_id]:
                dico[doc_id][valeur_parti] = []

        if texte is not None:
            for content in texte:
                if content.text is not None:
                    dico[doc_id][valeur_parti] += tokenizer_and_normalizer(content.text)


    return dico

def parse_xml_test(file, dico):
    tree = ET.parse(file)
    root = tree.getroot()
    langue = file.split('_')[-1][:2]
    ref_file = '..\deft09\Donnée_de_référence\deft09_parlement_ref_'+langue+'.txt'
    for i, doc in enumerate(root.findall('.//doc')):
        texte = doc.find('.//texte')
        doc_id = doc.get('id')
        doc_id = (doc_id, langue)
        dico[doc_id] = {}

        with open(ref_file, 'r', encoding='UTF-8') as f:
            for line in f:
                l = line.split()
                if l[0] == doc_id[0]:
                    parti = line.split()[1].strip() if len(line.split()) > 1 else None

        if parti is not None and parti not in dico[doc_id]:
            dico[doc_id][parti] = []

        if texte is not None:
            for content in texte:
                if content.text is not None:
                    dico[doc_id][parti] += tokenizer_and_normalizer(content.text)

    return dico

def create_df(dico):
    print("writing dic")
    counter = 0
    with open('../output/dataframe_train.csv', 'w', encoding='UTF8') as file:
        writer=csv.writer(file)
        header = ['doc', 'parti', 'texte']
        writer.writerow(header)
        for cle, valeur in dico.items():
            writer.writerow
            number = cle
            for parti, txt in valeur.items():
                data = [number, parti, txt]
            writer.writerow(data)

def create_df_test(dico):
    print("writing dic")
    counter = 0
    with open('../output/dataframe_test.csv', 'w', encoding='UTF8') as file:
        writer=csv.writer(file)
        header = ['doc', 'parti', 'texte']
        writer.writerow(header)
        for cle, valeur in dico.items():
            writer.writerow
            number = cle
            for parti, txt in valeur.items():
                data = [number, parti, txt]
            writer.writerow(data)

dico_train = {}
for file in train_files:
    print("parsing", file)
    dico_train.update(parse_xml_train(file, dico_train))
#create_df(dico_train)

dico_test = {}
print(test_files)
for file in test_files:
    print("parsing", file)
    dico_test.update(parse_xml_test(file, dico_test))
#create_df_test(dico_test)

from collections import Counter

def get_counts(doc):
    "compte du nombre de mots par document"
    for parti, content in doc.items():
        print(content)
        return Counter(content)

bows = [get_counts(dico_train[doc]) for doc in dico_train]

#recuper tout le vocabulaire
vocab = sorted(set().union(*bows))
