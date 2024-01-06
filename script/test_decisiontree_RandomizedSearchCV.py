# script pour utiliser les meilleurs paramètres trouvés avec RandomizedSearchCV sur les données de test
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# les chemins d'accès
train_data_path = r'C:\Users\alien\OneDrive\Documents\classifieursBIS\classifieurs\dataframe_train_without_stopwords.csv'
test_data_path = r'C:\Users\alien\OneDrive\Documents\classifieursBIS\classifieurs\dataframe_test_without_stopwords.csv'

# les données d'apprentissage
df_train = pd.read_csv(train_data_path)
texts_train = df_train['content'].apply(lambda x: ' '.join(eval(x)))
parties_train = df_train['parti']

# les données de test
df_test = pd.read_csv(test_data_path)
texts_test = df_test['content'].apply(lambda x: ' '.join(eval(x)))
parties_test = df_test['parti']

# construction du pipeline avec les meilleurs paramètres trouvés
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.7733551396716398, 
        min_df=5, 
        max_features=12989, 
        ngram_range=(1, 1)
    )),
    ('svd', TruncatedSVD(n_components=103)),
    ('clf', DecisionTreeClassifier(
        max_depth=12, 
        min_samples_split=7, 
        min_samples_leaf=2, 
        random_state=42
    )),
])

# entraînement du pipeline sur l'ensemble des données d'entraînement
pipeline.fit(texts_train, parties_train)

# prédictions sur les données de test
y_pred_test = pipeline.predict(texts_test)

# évaluation des performances sur les données de test
print(classification_report(parties_test, y_pred_test))
