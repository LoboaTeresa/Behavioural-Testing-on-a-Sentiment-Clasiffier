import sys
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

sys.path.append('../src')  # Assuming 'process_data.py' is in the 'src' directory
from src.pre_process_data import pre_process_data

def train_models(train_csv_path: str = '../data/train.csv') -> list:
    __, r_train_X, r_train_y = pre_process_data(train_csv_path , data_type = 'train')

    models = [RandomForestClassifier(max_depth = 15, n_estimators = 3000, random_state = 2),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(random_state = 2, max_iter = 500),
            KMeans(n_clusters = 4, init = 'k-means++') ]

    for m in models:
        m.fit(r_train_X, r_train_y)
        m_name = m.__class__.__name__
        with open(f'model_{m_name}.pkl','wb') as f:
            pickle.dump(m,f)
    return models
