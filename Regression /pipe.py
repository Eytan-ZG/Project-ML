#%%
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# --- utilitaire ---
def json_to_list(s, key='name'):
    if pd.isna(s) or s == '[]':
        return []
    try:
        return [d[key] for d in ast.literal_eval(s)]
    except:
        return []

# --- Transformers ---
class ReleaseYearBinner(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='release_date'):
        self.date_col = date_col
        self.bins = [1920, 1940, 1960, 1980, 2000, np.inf]  
        self.labels = [5, 4, 3, 2, 1]

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        X = X.copy()
        X['release_year'] = X[self.date_col].str[:4].astype(float)
        X['release_period_score'] = pd.cut(
            X['release_year'], bins=self.bins, labels=self.labels, right=False
        ).astype(float)
        return X

class LanguageGroupMapper(BaseEstimator, TransformerMixin):
    def __init__(self, lang_col='original_language'):
        self.lang_col = lang_col
        self.groups = [
            ['te','id','he','fa','ar','nl'], ['da','xx','pl','sv','ja','it'],
            ['af','el','is'], ['nb','ko','es'],
            ['hu','cn','fr','pt','ru','de','ps','zh'],
            ['no','en','hi','sl','th','ta'],
            ['vi','tr','ro'], ['ky']
        ]
        self.group_scores = {i: 8-i for i in range(8)}  

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        X = X.copy()
        def map_language(lang):
            for i, grp in enumerate(self.groups):
                if lang in grp:
                    return self.group_scores[i]
            return 0
        X['lang_group_score'] = X[self.lang_col].apply(map_language)
        return X

class CountryVoteBinnerMulti(BaseEstimator, TransformerMixin):
    """Pipeline-safe numeric feature based on target"""
    def __init__(self, thresholds=[5,6.5], scores=[1,2,3]):
        self.thresholds = thresholds
        self.scores = scores
        self.y_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for CountryVoteBinnerMulti")
        self.y_ = y
        return self

    def transform(self, X):
        X = X.copy()
        indices = np.digitize(self.y_, bins=self.thresholds)
        X['country_vote_score'] = np.array(self.scores)[indices]
        return X

class ProductionCompanyTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, prod_col='production_companies', json_key='name'):
        self.prod_col = prod_col
        self.json_key = json_key
        self.company_stats_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for ProductionCompanyTargetEncoder")
        X_ = X.copy()
        X_['_prod_list'] = X_[self.prod_col].apply(lambda s: json_to_list(s, key=self.json_key))
        X_exploded = X_.explode('_prod_list')
        X_exploded['target'] = y.reindex(X_exploded.index)
        self.company_stats_ = X_exploded.groupby('_prod_list')['target'].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X['_prod_list'] = X[self.prod_col].apply(lambda s: json_to_list(s, key=self.json_key))
        def mean_company_rating(companies):
            if len(companies) == 0:
                return None
            vals = self.company_stats_.loc[self.company_stats_.index.intersection(companies)]
            return vals.mean() if len(vals) > 0 else None
        X['prod_company_mean_rating'] = X['_prod_list'].apply(mean_company_rating)
        return X.drop(columns=['_prod_list'])

class PopularityVoteInteraction(BaseEstimator, TransformerMixin):
    def __init__(self, vote_col='vote_count', pop_col='popularity'):
        self.vote_col = vote_col
        self.pop_col = pop_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['popularityXvote_count_sqrd'] = (X[self.vote_col] ** 2) * X[self.pop_col]
        return X

class BudgetRuntimeInteraction(BaseEstimator, TransformerMixin):
    def __init__(self, budget_col='budget', runtime_col='runtime'):
        self.budget_col = budget_col
        self.runtime_col = runtime_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['budgetXruntime'] = (X[self.budget_col] ** 2) * X[self.runtime_col]
        return X

class GenreMultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, genre_col='genres', json_key='name', drop_genres=None):
        self.genre_col = genre_col
        self.json_key = json_key
        self.drop_genres = drop_genres if drop_genres else []
        self.mlb = None
        self.classes_ = None
    def fit(self, X, y=None):
        genres_list = X[self.genre_col].apply(lambda s: json_to_list(s, key=self.json_key))
        genres_list = genres_list.apply(lambda g: [x for x in g if x not in self.drop_genres])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(genres_list)
        self.classes_ = self.mlb.classes_
        return self
    def transform(self, X):
        X = X.copy()
        genres_list = X[self.genre_col].apply(lambda s: json_to_list(s, key=self.json_key))
        genres_list = genres_list.apply(lambda g: [x for x in g if x not in self.drop_genres])
        genres_encoded = self.mlb.transform(genres_list)
        df_genres = pd.DataFrame(genres_encoded, columns=self.classes_, index=X.index)
        return pd.concat([X, df_genres], axis=1)

class KeywordKMeansClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_col='keywords', json_key='name', top_k=30, n_clusters=6, random_state=42):
        self.keyword_col = keyword_col
        self.json_key = json_key
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.top_keywords_ = None
        self.mlb_ = None
        self.kmeans_ = None
    def fit(self, X, y=None):
        keywords_list = X[self.keyword_col].apply(lambda s: json_to_list(s, key=self.json_key))
        all_keywords = [kw for kws in keywords_list for kw in kws]
        keyword_counts = Counter(all_keywords)
        self.top_keywords_ = [kw for kw,_ in keyword_counts.most_common(self.top_k)]
        keywords_filtered = keywords_list.apply(lambda kws: [kw for kw in kws if kw in self.top_keywords_])
        self.mlb_ = MultiLabelBinarizer(classes=self.top_keywords_)
        X_kw = self.mlb_.fit_transform(keywords_filtered)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X_kw)
        return self
    def transform(self, X):
        X = X.copy()
        keywords_list = X[self.keyword_col].apply(lambda s: json_to_list(s, key=self.json_key))
        keywords_filtered = keywords_list.apply(lambda kws: [kw for kw in kws if kw in self.top_keywords_])
        X_kw = self.mlb_.transform(keywords_filtered)
        X['cluster_kmeans'] = self.kmeans_.predict(X_kw)
        return X

# --- Pipeline finale ---
numeric_features = [
    'budget','runtime','vote_count','popularity',
    'release_period_score','lang_group_score',
    'country_vote_score','prod_company_mean_rating',
    'popularityXvote_count_sqrd','budgetXruntime'
]
categorical_features = ['cluster_kmeans']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

full_pipeline = Pipeline([
    ('release_year', ReleaseYearBinner()),
    ('language_group', LanguageGroupMapper()),
    ('country_group', CountryVoteBinnerMulti()),
    ('prod_company_te', ProductionCompanyTargetEncoder()),
    ('pop_vote_inter', PopularityVoteInteraction()),
    ('budget_runtime_inter', BudgetRuntimeInteraction()),
    ('genres_ohe', GenreMultiLabelEncoder(
        drop_genres=['Mystery','Documentary','Foreign','Family','TV Movie','Western','Fantasy'])),
    ('keywords_cluster', KeywordKMeansClusterer()),
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))
])

# --- Utilisation ---
dataset = pd.read_csv('tmdb_5000_movies.csv', index_col='id')
dataset.columns = dataset.columns.str.strip()  

X = dataset.drop(columns=['vote_average'])
y = dataset['vote_average']

full_pipeline.fit(X, y)
y_pred = full_pipeline.predict(X)

# %%
