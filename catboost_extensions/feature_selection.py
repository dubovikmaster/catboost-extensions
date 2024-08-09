import math
import pickle
import random
from collections import defaultdict

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from catboost import Pool, CatBoostClassifier, CatBoostRegressor

import shap

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics._scorer import check_scoring
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

from .utils import send_msgs_to_telegram


class FeatureSelectorMixin:
    def __init__(
            self,
            estimator,
            estimator_parameters=None,
            telegram_api_token=None,
            telegram_chat_id=None,
            cv=None,
    ):
        self.estimator = estimator
        self.telegram_api_token = telegram_api_token
        self.telegram_chat_id = telegram_chat_id
        self.send_msgs_to_tg = self._check_send_to_telegram()
        self.cv = cv
        if self.cv is None:
            if isinstance(self.estimator, CatBoostClassifier):
                self.cv = StratifiedKFold(5, shuffle=True, random_state=2023)
            else:
                self.cv = KFold(5, shuffle=True, random_state=2023)
        elif isinstance(self.cv, int):
            if isinstance(self.estimator, CatBoostClassifier):
                self.cv = StratifiedKFold(self.cv, shuffle=True, random_state=2023)
            else:
                self.cv = KFold(self.cv, shuffle=True, random_state=2023)
        self.cat_features_ = None
        self.text_features_ = None
        self.parameters = self._prepare_parameters(estimator_parameters)

    def _check_send_to_telegram(self):
        send_msgs_to_tg = False
        if self.telegram_chat_id is not None and self.telegram_api_token is not None:
            send_msgs_to_tg = True
        return send_msgs_to_tg

    def _prepare_parameters(self, parameters):
        if parameters is None:
            copy_parameters = {}
        else:
            copy_parameters = dict(parameters)
        try:
            self.cat_features_ = copy_parameters.pop('cat_features')
            self.text_features_ = copy_parameters.pop('text_features')
        except KeyError:
            pass
        return copy_parameters

    def _cat_and_text_idx_prepare(self, x, features_type='cat'):
        if features_type == 'cat':
            feat_idx = list(self.cat_features_)
            cols = list(self.cat_features_)
        else:
            feat_idx = list(self.text_features_)
            cols = list(self.text_features_)
        if isinstance(x, pd.DataFrame):
            for idx, c in enumerate(cols):
                if isinstance(c, str):
                    feat_idx[idx] = x.columns.get_loc(c)
        return feat_idx

    def _recalculate_cat_and_text_idx(self, candidate_mask, features_type='cat'):
        features = self.cat_features_ if features_type == 'cat' else self.text_features_
        null_cnt = 0
        new_cat_idx = None
        if features is not None:
            new_cat_idx = list()
            for idx, e in enumerate(candidate_mask):
                if e == 0:
                    null_cnt += 1
                elif idx in features:
                    new_cat_idx.append(idx - null_cnt)
        return new_cat_idx

    @staticmethod
    def _get_data_subsample(X, indices):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.iloc[indices]
        return X[indices]

    def _split_data(self, X, y, train_idx, test_idx):
        X_train = self._get_data_subsample(X, train_idx)
        X_test = self._get_data_subsample(X, test_idx)
        y_train = self._get_data_subsample(y, train_idx)
        y_test = self._get_data_subsample(y, test_idx)
        return X_train, X_test, y_train, y_test

    def save(self, file_name):
        with open(f'{file_name}.pkl', 'wb') as f:  # open a text file
            pickle.dump(self.__dict__, f)

    def load(self, file_name):
        with open(f'{file_name}.pkl', 'rb') as f:
            self.__dict__ = pickle.load(f)


class CatboostSequentialFeatureSelector(FeatureSelectorMixin):
    def __init__(self, estimator, estimator_parameters=None, n_features_to_select=1, direction='forward', scoring=None,
                 cv=None, show_progress=True, show_progress_per_features=False, telegram_api_token=None,
                 telegram_chat_id=None, tolerance=None, verbose=False):
        super().__init__(
            estimator,
            estimator_parameters,
            telegram_api_token,
            telegram_chat_id,
            cv
        )
        self.direction = direction
        self.scoring = scoring
        self.n_features_to_select = n_features_to_select
        self.show_progress = show_progress
        self.show_progress_per_features = show_progress_per_features
        self.tolerance = tolerance
        self.verbose = verbose
        self.selected_features_with_score_ = {}
        self.finished_steps_ = 0
        self.last_score_ = -np.inf
        self.support_ = None
        self.support_features_names_ = None
        self.dropped_features_ = None
        self.dropped_features_names_ = None

    def fit(self, X, y, n_features_to_select=None):
        if n_features_to_select is not None:
            self.n_features_to_select = n_features_to_select
        n_features = X.shape[1]
        if self.cat_features_ is not None:
            self.cat_features_ = self._cat_and_text_idx_prepare(X)
        if self.text_features_ is not None:
            self.text_features_ = self._cat_and_text_idx_prepare(X, features_type='text')
        if self.support_ is None:
            current_mask = np.zeros(shape=n_features, dtype=bool)
        else:
            current_mask = self.support_
        n_iterations = (
            self.n_features_to_select if self.direction == "forward"
            else n_features - self.n_features_to_select
        )
        old_score = self.last_score_
        try:
            for i in tqdm(range(self.finished_steps_, n_iterations), disable=not self.show_progress, desc='STEPS'):
                new_feature_idx, new_score = self._get_best_new_feature_score(
                    X, y, current_mask
                )
                if self.tolerance is not None and (new_score - old_score) < self.tolerance:
                    break
                old_score = new_score
                selected_features = new_feature_idx
                if isinstance(X, pd.DataFrame):
                    selected_features = X.columns[new_feature_idx]
                if self.direction == 'forward':
                    msgs = f'ITERATIONS:{i + 1} \n\t New score is {old_score:.4f} \n\t ' \
                           f'Selected feature: {selected_features}'
                else:
                    msgs = f'ITERATIONS:{i + 1} \n\t New score is {old_score:.4f} \n\t ' \
                           f'Removed feature: {selected_features}'
                if self.verbose:
                    print(msgs)
                if self.send_msgs_to_tg:
                    send_msgs_to_telegram(
                        msgs,
                        self.telegram_api_token,
                        self.telegram_chat_id
                    )
                current_mask[new_feature_idx] = True
                self.selected_features_with_score_[new_feature_idx] = old_score
                self.finished_steps_ += 1
                self.last_score_ = old_score
        except Exception as e:
            print(f'Exception occurred while running step {i}: {e}')
        finally:
            if self.direction == "backward":
                current_mask = ~current_mask
            self.support_ = current_mask
            self.dropped_features_ = ~current_mask
            if isinstance(X, pd.DataFrame):
                self.support_features_names_ = X.columns[self.support_]
                self.dropped_features_names_ = X.columns[self.dropped_features_]
        return self

    def _get_best_new_feature_score(self, X, y, current_mask):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in tqdm(candidate_feature_indices, disable=not self.show_progress_per_features,
                                desc='FEATURES COUNT'):
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            X_new = X.iloc[:, candidate_mask] if isinstance(X, pd.DataFrame) else X[:, candidate_mask]
            new_cat_idx = self._recalculate_cat_and_text_idx(candidate_mask)
            new_text_idx = self._recalculate_cat_and_text_idx(candidate_mask, features_type='text')
            estimator = self.estimator(cat_features=new_cat_idx,
                                       text_features=new_text_idx,
                                       **self.parameters).copy()
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=self.cv,
                scoring=self.scoring,
                error_score=-np.inf
            ).mean()
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        return new_feature_idx, scores[new_feature_idx]

    def show_scores(self, **kwargs):
        if len(self.selected_features_with_score_) == 0:
            raise ValueError('You must run "fit" before.')
        features_indices = np.flatnonzero(self.support_)
        col = 'Selected feature'
        feature_names = self.support_features_names_
        title = 'Selected features per steps'
        if self.direction == 'backward':
            col = 'Removed features'
            title = 'Removed features per steps'
            features_indices = np.flatnonzero(self.dropped_features_)
            feature_names = self.dropped_features_names_
        df = pd.DataFrame(self.selected_features_with_score_.items(),
                          columns=[col, 'score'])
        df['step'] = list(range(df.shape[0]))
        if self.support_features_names_ is not None:
            mapping = dict(zip(features_indices, feature_names))
            df[col] = df[col].map(mapping)
        if df.score.max() < 0:
            df.score = -df.score
        fig = px.line(df, x='step', y='score', text=col, title=title,
                      labels={'step': 'Step', 'score': 'Score'}, **kwargs)
        fig.update_traces(textposition="bottom left")
        fig.show()


class CVPermutationImportance(FeatureSelectorMixin):
    def __init__(self, estimator, estimator_parameters=None, scoring=None, cv=None, show_progress=True,
                 telegram_api_token=None, telegram_chat_id=None, verbose=False, n_repeats=5,
                 n_jobs=None, random_state=None, use_test_data_for_evaluation=False
                 ):
        super().__init__(telegram_api_token, telegram_chat_id)
        self.scoring = scoring
        self.parameters = estimator_parameters if estimator_parameters is not None else {}
        self.cv = cv
        self.show_progress = show_progress
        self.verbose = verbose
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_test_data_for_evaluation = use_test_data_for_evaluation
        self.result_ = None
        self.sorted_importance_features_ = None
        self._sorted_importance_idx = None

    def fit(self, X, y):
        result_dict = {'importance': np.zeros((X.shape[1], self.n_repeats))}
        for (train_idx, test_idx) in tqdm(self.cv.split(X, y), total=self.cv.n_splits,
                                          disable=not self.show_progress, desc='CV'):
            model = self.estimator(**self.parameters)
            X_train, X_test, y_train, y_test = self._split_data(X, y, train_idx, test_idx)
            if self.use_test_data_for_evaluation:
                eval_set = (X_test, y_test)
            else:
                eval_set = None
            model.fit(X_train,
                      y_train,
                      eval_set=eval_set,
                      )
            result = permutation_importance(model,
                                            X_test, y_test,
                                            scoring=self.scoring,
                                            n_repeats=self.n_repeats,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state,
                                            )
            result_dict['importance'] += result['importance']
        result_dict['importance'] /= self.cv.n_splits
        result_dict['importance_mean'] = result_dict['importance'].mean(axis=1)
        result_dict['importance_std'] = result_dict['importance'].std(axis=1)
        self.result_ = result_dict
        sorted_importance_idx = self.result_['importance_mean'].argsort()
        self._sorted_importance_idx = sorted_importance_idx
        if isinstance(X, pd.DataFrame):
            self.sorted_importance_features_ = X.columns[sorted_importance_idx]
        else:
            self.sorted_importance_features_ = sorted_importance_idx

        return self

    def show_scores(self, **kwargs):
        if self.result_ is None:
            raise ValueError('You must run "fit" before.')

        importance = pd.DataFrame(
            self.result_['importance'][self._sorted_importance_idx].T,
            columns=self.sorted_importance_features_
        )
        fig = px.box(importance, title='Permutation importance', orientation='h',
                     labels={'variable': 'Features', 'value': "Decrease score"}, **kwargs)
        fig.add_vline(0, line_width=3, line_dash="dash", line_color="red")
        fig.show()


class CatboostCVRFE(FeatureSelectorMixin):
    def __init__(
            self,
            estimator,
            estimator_parameters=None,
            n_features_to_select=1,
            cv=None, step=1,
            scoring=None,
            show_progress=True,
            show_progress_per_features=False,
            telegram_api_token=None,
            telegram_chat_id=None,
            verbose=False,
            steps=1,
            importance_getter=None,
            sample_weight=None,
            weight_columns=None,
            use_test_data_for_evaluation=False,
    ):
        super().__init__(estimator, estimator_parameters, telegram_api_token, telegram_chat_id, cv)
        self.scoring = scoring
        self.steps = steps
        self._random_states = [self.cv.random_state]
        if self.steps > 1:
            self.cv.shuffle = True
            self._random_states = [random.randint(0, 2023) for _ in range(self.steps)]
        self.importance_getter = importance_getter
        self.sample_weight = sample_weight
        self.weight_columns = weight_columns
        self.dropped_features_names_ = None
        self.support_features_names_ = None
        self.sorted_importance_features_ = None
        self._sorted_importance_idx = None
        self.ranking_ = None
        self.support_ = None
        self.n_features_ = None
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.show_progress = show_progress
        self.show_progress_per_features = show_progress_per_features
        self.verbose = verbose
        self.scores = None
        self.use_test_data_for_evaluation = use_test_data_for_evaluation
        self._total_features = None

        if callable(scoring):
            self.scorer = scoring
        elif scoring is None or isinstance(scoring, str):
            self.scorer = check_scoring(estimator, scoring=scoring)

    def fit(self, X, y):
        self.scores = defaultdict(list)
        if self.cat_features_ is not None:
            self.cat_features_ = self._cat_and_text_idx_prepare(X)
        if self.text_features_ is not None:
            self.text_features_ = self._cat_and_text_idx_prepare(X, features_type='text')
        n_features = X.shape[1]
        self._total_features = n_features
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.zeros(n_features, dtype=int)
        n_iterations = math.ceil((n_features - self.n_features_to_select) / self.step) + 1
        # Elimination
        for i in tqdm(range(n_iterations), disable=not self.show_progress, desc='ITERATIONS'):
            X_new = X.iloc[:, support_] if isinstance(X, pd.DataFrame) else X[:, support_]
            new_cat_idx = self._recalculate_cat_and_text_idx(support_)
            new_text_idx = self._recalculate_cat_and_text_idx(support_, features_type='text')
            importance = np.zeros(np.sum(support_))
            scores_per_fold = list()
            for j in range(self.steps):
                self.cv.random_state = self._random_states[j]
                for (train_idx, test_idx) in self.cv.split(X, y):
                    estimator = self.estimator(cat_features=new_cat_idx,
                                               text_features=new_text_idx,
                                               **self.parameters).copy()
                    X_train, X_test, y_train, y_test = self._split_data(X_new, y, train_idx, test_idx)
                    if self.use_test_data_for_evaluation:
                        eval_set = (X_test, y_test)
                    else:
                        eval_set = None
                    if self.sample_weight is None:
                        estimator.fit(X_train,
                                      y_train,
                                      eval_set=eval_set,
                                      )
                    else:
                        sample_weight = compute_sample_weight(class_weight='balanced', y=X_train[self.weight_columns])
                        train_data_weight = Pool(
                            data=X_train,
                            label=y_train,
                            weight=sample_weight,
                            cat_features=new_cat_idx,
                        )
                        estimator.fit(train_data_weight,
                                      eval_set=eval_set,
                                      )
                    scores_per_fold.append(self.scorer(estimator, X_test, y_test))
                    if self.importance_getter is None:
                        importance += estimator.feature_importances_
                    elif self.importance_getter == 'shap':
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(X_test)
                        importance += np.abs(shap_values).mean(axis=0)
            self.scores['mean_score'].append(np.mean(scores_per_fold))
            self.scores['std_score'].append(np.std(scores_per_fold))
            importance = importance / (self.cv.n_splits * self.steps)
            step = min(self.step, np.sum(support_) - self.n_features_to_select)
            dropped_features_idx = support_.nonzero()[0][importance.argsort()[:step]]
            if step:
                if isinstance(X, pd.DataFrame):
                    msgs = f'ITERATIONS: {i + 1} \n\t Removed features: {list(X.columns[dropped_features_idx])}'
                else:
                    msgs = f'ITERATIONS: {i + 1} \n\t Removed features: {dropped_features_idx}'
                msgs = msgs + f'\n\t New score is {self.scores["mean_score"][-1]:.4f}'
                if self.verbose:
                    print(msgs)
                if self.send_msgs_to_tg:
                    send_msgs_to_telegram(
                        msgs,
                        self.telegram_api_token,
                        self.telegram_chat_id
                    )
                support_[dropped_features_idx] = False
                ranking_[~np.logical_not(support_)] += 1
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        sorted_importance_idx = self.ranking_.argsort()
        self._sorted_importance_idx = sorted_importance_idx
        if isinstance(X, pd.DataFrame):
            self.sorted_importance_features_ = X.columns[sorted_importance_idx]
        else:
            self.sorted_importance_features_ = sorted_importance_idx
        self.support_features_names_ = X.columns[self.support_] if isinstance(X, pd.DataFrame) else range(n_features)[
            self.support_]
        self.dropped_features_names_ = X.columns[~self.support_] if isinstance(X, pd.DataFrame) else range(n_features)[
            ~self.support_]
        return self

    def show_ranks(self, **kwargs):
        if self.ranking_ is None:
            raise ValueError('You must run "fit" before.')

        importances = pd.DataFrame(
            self.ranking_[self._sorted_importance_idx].reshape(-1, 1),
            index=self.sorted_importance_features_
        )
        fig = px.bar(importances, title='CVRFE importances', orientation='h',
                     labels={'variable': 'Features', 'value': "Importance scores"}, **kwargs)
        fig.show()

    def show_scores(self, **kwargs):
        if self.scores is None:
            raise ValueError('You must run "fit" before.')
        x = [i * self.step for i in range(len(self.scores['mean_score']) - 1)]
        x = x + [self._total_features - self.n_features_to_select]
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=self.scores['mean_score'],
            error_y=dict(type='data',
                         array=self.scores['std_score'],
                         visible=True,
                         )
        ))
        fig.update_layout(xaxis_title='Dropped features count', yaxis_title='Score', title='CVRFE scores', **kwargs)
        fig.add_scatter(x=[x[np.argmax(self.scores['mean_score'])]],
                        y=[np.max(self.scores['mean_score'])],
                        mode='markers',
                        marker=dict(color='red',
                                    size=10
                                    ),
                        name='Best score',
                        )
        fig.show()
