# script pour la recherche des meilleurs paramètres par grille pour le classifieur Decision Tree
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import classification_report

# les données d'apprentissage
df_train = pd.read_csv(r'C:\...\Documents\test\dataframe_train.csv')
# prétraitement du texte : transformation de listes de mots en chaînes de caractères pour la vectorisation
texts_train = df_train['texte'].apply(lambda x: ' '.join(eval(x)))
# extraction des étiquettes associées au texte
parties_train = df_train['parti']

# pipeline de traitement
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),# vectorisation TF-IDF du texte
    ('svd', TruncatedSVD()), # réduction de dimensionnalité avec SVD
    ('clf', DecisionTreeClassifier(random_state=42)), # classification avec Decision Tree
])

# paramètres pour la recherche sur grille 
parameters = {
    'tfidf__max_df': [0.5, 0.75],# paramètres pour TF-IDF
    'tfidf__min_df': [1, 5],
    'tfidf__max_features': [None, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)], # utilisation des unigrammes et bigrammes
    'svd__n_components': [50, 100], # nombre de composants pour SVD
    'clf__max_depth': [None, 50], # profondeur maximale de l'arbre de décision
    'clf__min_samples_split': [2, 5], # nombre minimum d'échantillons requis pour diviser un nœud
    'clf__min_samples_leaf': [1, 2], # nombre minimum d'échantillons requis pour être à une feuille
}

# configuration de la validation croisée 
kf = KFold(n_splits=3, shuffle=True, random_state=42) 

# création du modèle GridSearchCV 
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=kf, n_jobs=2, verbose=2)

# exécution de la recherche sur grille
grid_search.fit(texts_train, parties_train)

# affichage des meilleurs paramètres et du meilleur score
print("Meilleurs paramètres trouvés :")
print(grid_search.best_params_)
print("Score avec ces paramètres :")
print(grid_search.best_score_)

# utilisation du meilleur modèle pour la classification
best_pipeline = grid_search.best_estimator_

# prédiction et évaluation avec le meilleur modèle
y_pred_cv = cross_val_predict(best_pipeline, texts_train, parties_train, cv=kf)
print(classification_report(parties_train, y_pred_cv))

# calcul et affichage de la précision globale
accuracy = (y_pred_cv == parties_train).mean()
print(f'Accuracy: {accuracy}')
