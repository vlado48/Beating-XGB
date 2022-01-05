import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

def apply_XGBc(X, y, split=True, params={}):
    if split:
        dataset_splits = split_data(X, y)
        model = get_XGBc_model(dataset_splits)        
    else: 
        dataset_splits = {'X_train': X, 'y_train': y}
        model = get_XGBc_model(dataset_splits, eval_set=False)
        return model, None
        
    report = evaluate_XGBc_model(model, dataset_splits['X_test'],
                                 dataset_splits['y_test'])    
    return model, report

def split_data(data_X, data_y):
    data_y = data_y    # Categories started at 1, needed decrement
    
    X, X_test, y, y_test = train_test_split(data_X, data_y,
                                            test_size=0.1, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                            test_size=0.2, random_state=0)
    
    print(f'\nTrain samples {len(y_train)},'
          f'\nValidation samples {len(y_valid)},',
          f'\nTest samples {len(y_test)}')
    
    return dict(X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                X_test=X_test, y_test=y_test)

def get_XGBc_model(datasplits, params={}, eval_set=True):
    def_params = {'n_estimators': 1000,
                  'max_depth': 5,
                  'learning_rate': 0.5,
                  'objective': 'multi:softmax',
                  'booster': 'gbtree',
                  'gamma' : 0,
                  'min_child_weight': 1,
                  'reg_alpha': 0,
                  'reg_lambda': 0,
                  'num_parallel_tree': 1,
                  'num_class': len(np.unique(datasplits['y_train']))
                  }
    
    for param in params:
        def_params[param] = params[param]
    
    model = XGBClassifier(**def_params)
    if eval_set:
        model.fit(datasplits['X_train'], datasplits['y_train'],
                  eval_set=[(datasplits['X_valid'], datasplits['y_valid'])],
                  early_stopping_rounds=10)
    else: 
        model.fit(datasplits['X_train'], datasplits['y_train'])    

    return model

def evaluate_XGBc_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
    return report

#%% Data loading / splitting
iris = datasets.load_iris(as_frame=True)
iris_model, iris_report = apply_XGBc(iris.data, iris.target)

#%%
wine = datasets.load_wine(as_frame=True)
wine_model, wine_report = apply_XGBc(wine.data, wine.target)

#%%
cancer = datasets.load_breast_cancer(as_frame=True)

label_encoder = LabelEncoder()
cancer_y = label_encoder.fit_transform(cancer.target.to_numpy())
cancer_X = pd.get_dummies(cancer.data)

cancer_model, cancer_report = apply_XGBc(cancer_X, cancer_y)
#%%
print((np.unique(cancer.target.to_numpy())))


#%% -------- Large datasets --------
# Forest cover type
covtype = datasets.fetch_covtype(as_frame=True)
covtype_splits = split_data(covtype.data, covtype.target-1)

params = {'n_estimators': 10000}
covtype_model = get_XGBc_model(covtype_splits, params)
report = evaluate_XGBc_model(covtype_model, covtype_splits['X_test'],
                             covtype_splits['y_test'])


#%% -------- Large datasets --------
# News group type
newsgroups = datasets.fetch_20newsgroups(remove=('headers',
                                                 'footers',
                                                 'quotes'))

vectorizer = TfidfVectorizer()

X_data = vectorizer.fit_transform(newsgroups.data)
y_data = newsgroups.target

newsgroups_params = {'max_depth' : 6}
newsgroups_model, newsgroups_report = apply_XGBc(X_data, y_data, params=params)

#%% -------- Large datasets --------
# Credit risk
risk = datasets.fetch_openml(data_id=31, as_frame=True)
apply_XGBc(risk.data, risk.target)

#%% Columns are no all numerical, would need encoding
kdd = datasets.fetch_kddcup99(subset='SA', as_frame=True)

enc = OrdinalEncoder()
kdd_y = enc.fit_transform(kdd.target.to_numpy().reshape(1, -1)).reshape(-1, 1)

apply_XGBc(kdd.data, kdd_y)