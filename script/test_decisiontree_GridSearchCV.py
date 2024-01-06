# script pour utiliser les meilleurs paramètres trouvés avec GridSearchCV sur les données de test
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# les chemins d'accès
chemin_train = r'C:\...\classifieurs\dataframe_train_without_stopwords.csv'
chemin_test = r'C:\...\classifieurs\dataframe_test_without_stopwords.csv'

# les données de train
df_train = pd.read_csv(chemin_train)
texts_train = df_train['content'].apply(lambda x: ' '.join(eval(x)))
parties_train = df_train['parti']

# les données de test
df_test = pd.read_csv(chemin_test)
texts_test = df_test['content'].apply(lambda x: ' '.join(eval(x)))
parties_test = df_test['parti']

# pipeline avec les meilleurs paramètres trouvés
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.75, min_df=5, ngram_range=(1, 2))),
    ('svd', TruncatedSVD(n_components=50)), 
    ('clf', DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)),
])

# pipeline sur l'ensemble des données d'entraînement
pipeline.fit(texts_train, parties_train)

# prédictions sur l'ensemble de données de test
y_pred_test = pipeline.predict(texts_test)

# les performances sur l'ensemble de test
print(classification_report(parties_test, y_pred_test))
