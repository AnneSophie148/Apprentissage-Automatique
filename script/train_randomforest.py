# script pour la recherche des meilleurs paramètres avec RandomizedSearchCV pour le classifieur Random Forest
import pandas as pd
from scipy.stats import randint as sp_randint, uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.metrics import classification_report

# les données d'apprentissage
df_train = pd.read_csv('/.../classifieurs/dataframe_train.csv')
# prétraitement du texte : transformation de listes de mots en chaînes de caractères pour la vectorisation
texts_train = df_train['texte'].apply(lambda x: ' '.join(eval(x)))
# extraction des étiquettes associées au texte
parties_train = df_train['parti']

# pipeline de traitement avec RandomForestClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), # réduction du nombre maximal de caractéristiques
    ('svd', TruncatedSVD()),  # réduction du nombre de composants SVD
    ('clf', RandomForestClassifier(random_state=42)),
])

# paramètres pour la recherche aléatoire
parameters = {
    'tfidf__max_df': uniform(0.5, 1), # la fréquence maximale dans les documents pour les termes ( on ignore les plus fréquents)
    'tfidf__min_df': sp_randint(1, 6), # fréquence minimale requise pour qu'un terme soit inclus
    'tfidf__max_features': sp_randint(5000, 20001),# limite du nombre total de caractéristiques sélectionnées
    'tfidf__ngram_range': [(1, 1), (1, 2)],# plage des n-grams à extraire, soit unigrams seulement, soit unigrams et bigrams
    'svd__n_components': sp_randint(50, 301), # nombre de composants à garder lors de l'utilisation de SVD
    'clf__n_estimators': sp_randint(100, 500),# nombre d'arbres dans la forêt
    'clf__max_depth': [None] + list(range(10, 101)),  # profondeur maximale des arbres
    'clf__min_samples_split': sp_randint(2, 11), # nombre minimal d'échantillons pour diviser un nœud
    'clf__min_samples_leaf': sp_randint(1, 5), # nombre minimal d'échantillons pour être à un nœud feuille
    'clf__max_features': ['sqrt', 'log2'],  # nombre de caractéristiques pour la meilleure division
    'clf__bootstrap': [True, False],  # utilisation de bootstrap pour la construction des arbres
}

# configuration de la validation croisée
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# création du modèle RandomizedSearchCV avec le pipeline et les paramètres
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

# prédiction et évaluation avec le meilleur modèle en utilisant la validation croisée
y_pred_cv = cross_val_predict(best_pipeline, texts_train, parties_train, cv=kf)

# évaluation des performances du modèle sur les données d'apprentissage
print(classification_report(parties_train, y_pred_cv))

# calcul de la précision globale
accuracy = (y_pred_cv == parties_train).mean()
print(f'Accuracy: {accuracy}')

