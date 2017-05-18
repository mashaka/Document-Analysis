#####
# Author: Maria Sandrikova
#####

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import time


def test_classifier(clf, X, y, n_splits, scoring):
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring=scoring, n_jobs=-1)
    return np.mean(scores)


def grid_search(clf, X, y, params, n_splits, scoring):
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scoring, n_jobs=-1, cv=n_splits)
    grid.fit(X, y)
    return grid.best_score_, grid.best_estimator_


def add_prefix_to_dict_keys(prefix, dict_raw):
    new_dict = dict()
    for key, value in dict_raw.items():
        new_dict[prefix + key] = value
    return new_dict


class ExtendedModel:
    
    def __init__(self, model, params=None):
        self.model = model
        self.params = params


class ClfTester:

    def __init__(self, feature_models, clfs, scoring, n_splits):
        self.feature_models = feature_models
        self.clfs = clfs
        self.scoring = scoring
        self.n_splits = n_splits

    
    def test(self, X, y, show_time_log=True, print_best=True, best_score=-1):
        best_score = -1
        best_clf_desc = 'None'
        clf_names = list(self.clfs.keys())
        features_names = list(self.feature_models.keys())
        df_scores = pd.DataFrame(columns=clf_names, index=features_names)
        df_estimators = pd.DataFrame(columns=clf_names, index=features_names)
        for clf_name in clf_names:
            for features_name in features_names:
                start_time = time.time()
                if show_time_log:
                    print('Start processing {} + {}'.format(clf_name, features_name))
                clf = self.clfs[clf_name]
                features = self.feature_models[features_name]
                if features.model is not None:
                    clf_full = Pipeline([
                        ('Features', features.model), 
                        ('Classifier', clf.model)
                    ])
                else:
                    clf_full = Pipeline([ 
                        ('Classifier', clf.model)
                    ])
                params = dict()
                if clf.params is not None:
                    params.update(add_prefix_to_dict_keys('Classifier__', clf.params))
                if features.params is not None:
                    params.update(add_prefix_to_dict_keys('Features__', features.params))
                if params is None:
                    score, best_estimator = test_classifier(clf_full, X, y, self.n_splits, self.scoring), clf_full
                else:
                    score, best_estimator = grid_search(clf_full, X, y, params, self.n_splits, self.scoring)
                df_scores[clf_name][features_name] = score
                df_estimators[clf_name][features_name] = best_estimator
                if show_time_log:
                    seconds = int(time.time() - start_time)
                    minutes, seconds = divmod(seconds, 60)
                    print('Finish in {}:{}s with score {}'.format(minutes, seconds, round(score, 4)))
                if score > best_score:
                    best_score = score
                    best_clf_desc = self.get_pretty_clf_info(
                        '{} + {}'.format(features_name, clf_name), 
                        score, best_estimator, params
                    )
        if print_best:
            print('--- Best model ---')
            print(best_clf_desc)
        return df_scores, df_estimators

    @staticmethod
    def get_pretty_clf_info(name, score, clf, params_dict):
        desc = '{} with score {}'.format(name, round(score, 4))
        if params_dict is not None:
            desc += ' and params:\n'
            for p_key, p_value in params_dict.items():
                desc += '\t{}: {}\n'.format(p_key, clf.get_params()[p_key])
        return desc

