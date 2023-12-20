import numpy as np
import xml.etree.ElementTree as ET
import csv

import os

def get_files(path):
    file_list = []
    for dirpath, dirs, files in os.walk(path):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith('.xml'):
                file_list.append(fname)
    return file_list

path_train = '../deft09/Corpus_d_apprentissage'
path_test = '../deft09/Corpus_de_test'
train_files = get_files(path_train)
test_files = get_files(path_test)

import re
def tokenizer_and_normalizer(s):
    # on vire les nombres, les signes de ponctuation
    return [w.lower() for w in re.split(r"\s|\W", s.strip()) if w and all(c.isalpha() for c in w)]


def parse_xml(file, dico):
    tree = ET.parse(file)
    root = tree.getroot()
    for i, doc in enumerate(root.findall('.//doc')):
        dico[i+1]= {}
        parti = doc.find('.//PARTI')
        texte = doc.find('.//texte')

        if parti is not None:
            valeur_parti = parti.get('valeur')
            if valeur_parti not in dico[i+1]:
                dico[i+1][valeur_parti] = []

        if texte is not None:
            for content in texte:
                if content.text is not None:
                    dico[i+1][valeur_parti] += tokenizer_and_normalizer(content.text)

   
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



dico_train = {}
for file in train_files:
    print("parsing", file)
    dico_train.update(parse_xml(file, dico_train))
    #input()
create_df(dico_train)


