# # script pour la recherche aléatoire des meilleurs paramètres pour le classifieur Decision Tree
import pandas as pd
from scipy.stats import randint as sp_randint, uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import classification_report

# les données d'apprentissage
df_train = pd.read_csv('/.../Documents/classifieurs/dataframe_train.csv')
# prétraitement du texte : transformation de listes de mots en chaînes de caractères pour la vectorisation
texts_train = df_train['texte'].apply(lambda x: ' '.join(eval(x)))
# extraction des étiquettes associées aux textes
parties_train = df_train['parti']

# pipeline de traitement
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), # vectorisation avec TF-IDF
    ('svd', TruncatedSVD()),  # réduction de dimension de dimensionnalité avec SVD
    ('clf', DecisionTreeClassifier(random_state=42)), # classification avec arbre de décision
])

# paramètres pour la recherche aléatoire
parameters = {
    'tfidf__max_df': uniform(0.5, 1), # un flottant pour la fréquence maximale de document
    'tfidf__min_df': sp_randint(1, 6), # un entier pour la fréquence minimale de document
    'tfidf__max_features': sp_randint(5000, 20001), # nombre maximum de caractéristiques 
    'tfidf__ngram_range': [(1, 1), (1, 2)], # plage pour les n-grams
    'svd__n_components': sp_randint(50, 301), # nombre de composants pour la SVD
    'clf__max_depth': [None] + list(range(10, 101)), # profondeur maximale de l'arbre
    'clf__min_samples_split': sp_randint(2, 11),  # nombre minimum d'échantillons requis pour diviser un nœud
    'clf__min_samples_leaf': sp_randint(1, 5), # nombre minimum d'échantillons requis à un nœud feuille
}

# configuration de la validation croisée  avec KFold pour une meilleure estimation de la performance
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# création du modèle RandomizedSearchCV 
random_search = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=10, cv=kf, n_jobs=-1, verbose=2, random_state=42)

# exécution de la recherche aléatoire
random_search.fit(texts_train, parties_train)

# affichage des meilleurs paramètres et du meilleur score
print("Meilleurs paramètres trouvés :")
print(random_search.best_params_)
print("Score avec ces paramètres :")
print(random_search.best_score_)

# utilisation du meilleur modèle pour la classification
best_pipeline = random_search.best_estimator_

# prédiction et évaluation avec le meilleur modèle
y_pred_cv = cross_val_predict(best_pipeline, texts_train, parties_train, cv=kf)
print(classification_report(parties_train, y_pred_cv))

# calcul et affichage de la précision globale
accuracy = (y_pred_cv == parties_train).mean()
print(f'Accuracy: {accuracy}')

