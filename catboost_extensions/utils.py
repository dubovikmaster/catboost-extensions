import logging
import _thread as thread
import threading
from collections import defaultdict
from functools import wraps
from typing import (
    Callable,
    Optional,
    List,
    Union,
)

import platform
import signal
import os

import pandas as pd
from numpy.typing import ArrayLike
import numpy as np

from optuna.trial import Trial
from optuna.exceptions import TrialPruned

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    KFold,
)
from sklearn import metrics
from sklearn.metrics._scorer import check_scoring

from catboost import (
    Pool,
    CatBoostRanker,
    CatBoostRegressor,
    CatBoostClassifier,
)

logger = logging.getLogger('utils')

CatBoostModel = Union[CatBoostRanker, CatBoostRegressor, CatBoostClassifier]


def make_scorer(model, x, y, score=None):
    iterations = model.get_param('iterations')
    if iterations is None:
        iterations = 1000
    if score is None:
        score = model.get_param('loss_function')
    return model.eval_metrics(
        Pool(x, y, text_features=model.get_param('text_features'), cat_features=model.get_param('cat_features'), ),
        score, ntree_start=iterations - 1)[score][-1]


def stop_function():
    if platform.system() == 'Windows':
        thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def stopit_after_timeout(s, raise_exception=True, exception=TimeoutError):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer = threading.Timer(s, stop_function)
            try:
                timer.start()
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                msg = f'function \"{func.__name__}\" took longer than {s} s.'
                if raise_exception:
                    raise exception(msg)
                result = msg
            finally:
                timer.cancel()
            return result

        return wrapper

    return actual_decorator


class CrossValidator:
    """
    CrossValidator class performs cross-validation for a given CatBoost model.

    This class supports advanced cross-validation mechanics for CatBoost models
    and provides utility functions for model evaluation, dataset slicing, and
    integration with Optuna for hyperparameter optimization. It handles both
    classification and regression tasks while offering flexibility in integrating
    with various cross-validator schemes.

    Parameters
    ----------
    model: CatBoostModel
        The CatBoost model to be cross-validated.
    data: Union[Pool, pd.DataFrame, ArrayLike]
        Input data for training and validation, can be a CatBoost Pool,
                pandas DataFrame, or an array-like structure.
    scoring: Union[str, List[str]]
        Scoring metric(s) to evaluate the model. Can be a string for a
        single metric or a list of strings for multiple metrics.
    y: Optional[Union[pd.Series, pd.DataFrame, ArrayLike]]
        Target variable for supervised learning tasks. Optional if data is a Pool.
    cv: Union[BaseCrossValidator, int]
        Cross-validator instance or the number of splits. If an integer, it uses KFold for regression models
        or StratifiedKFold for classification models.
    weight_column:  Optional[ArrayLike]
        Sample weights applied to data samples. Optional.
    optuna_trial: Optional[Trial]
        Optuna Trial instance used for integration with hyperparameter optimization. Optional.
    n_folds_start_prune: Optional[int]
        Number of folds completed before starting Optuna pruning. Optional.
    group_id: Optional[ArrayLike]
        Array of group IDs for multi-group feature settings. Optional.
    subgroup_id: Optional[ArrayLike]
        Array of subgroup IDs for subgroup-specific settings. Optional.
    """

    def __init__(self, model: CatBoostModel, data: Union[Pool, pd.DataFrame, ArrayLike],
                 scoring: Union[str, List[str], dict[str, Callable]],
                 y: Optional[Union[pd.Series, pd.DataFrame, ArrayLike]] = None, cv: Union[BaseCrossValidator, int] = 5,
                 weight_column: Optional[ArrayLike] = None, optuna_trial: Optional[Trial] = None,
                 n_folds_start_prune: Optional[int] = None, group_id: Optional[ArrayLike] = None,
                 subgroup_id: Optional[ArrayLike] = None
                 ):
        self.model = model
        self.data = data
        self.y = y
        self._catboost_scoring = self.get_catboost_scores(scoring)
        self._sklearn_scores = self._get_sklearn_scores(scoring)
        self.cv = self._check_cv(cv, self.model)
        self.weight_column = weight_column
        self.optuna_trial = optuna_trial
        self.n_folds_start_prune = n_folds_start_prune
        self.group_id = group_id
        self.subgroup_id = subgroup_id

    @staticmethod
    def get_catboost_scores(scoring):
        if isinstance(scoring, str):
            scoring = [scoring]
        if not isinstance(scoring, dict):
            return [i for i in scoring if i not in metrics.get_scorer_names()]

    def _get_sklearn_scores(self, scoring):
        if isinstance(scoring, str):
            scoring = [scoring]
        if isinstance(scoring, dict):
            return check_scoring(self.model, scoring)
        if isinstance(scoring, list):
            sklearn_score = [i for i in scoring if i in metrics.get_scorer_names()]
            if sklearn_score:
                return check_scoring(self.model, sklearn_score)

    @staticmethod
    def _check_cv(cv: Union[int, BaseCrossValidator], model: CatBoostModel) -> BaseCrossValidator:
        if isinstance(cv, int):
            if isinstance(model, CatBoostRegressor):
                cv = KFold(5)
            else:
                cv = StratifiedKFold(5)
        elif isinstance(cv, BaseCrossValidator):
            return cv
        else:
            raise ValueError('cv must be int or BaseCrossValidator instance')

        return cv

    def get_model_iterations(self, cb_model: CatBoostModel) -> int:
        iterations = cb_model.get_param('iterations')
        if iterations is None:
            iterations = 1000
        return iterations

    def eval_model(self, cb_model: CatBoostModel, val_pool: Pool, metrics: str) -> float:
        score = cb_model.eval_metrics(val_pool, metrics=metrics, ntree_start=self.get_model_iterations(cb_model) - 1)
        return {key: val[0] for key, val in score.items()}

    def make_pool_slice(self, pool: Pool, idx: ArrayLike) -> Pool:
        pool_slice = pool.slice(idx)
        if self.weight_column is not None:
            weights = compute_sample_weight('balanced', y=self.weight_column[idx])
            pool_slice.set_weight(weights)
        if self.group_id is not None:
            pool_slice.set_group_id(self.group_id[idx])
        if self.subgroup_id is not None:
            pool_slice.set_subgroup_id(self.subgroup_id[idx])
        return pool_slice

    def fit(self) -> dict:
        if not isinstance(self.data, Pool):
            pool = Pool(
                self.data,
                self.y,
                text_features=self.model.get_param('text_features'),
                cat_features=self.model.get_param('cat_features'),
            )
        else:
            pool = self.data
        splits = self.cv.split(range(pool.shape[0]), self.y)
        scoring_dict = defaultdict(list)
        for idx, (train_idx, test_idx) in enumerate(splits):
            model = self.model.copy()
            train_pool = self.make_pool_slice(pool, train_idx)
            test_pool = self.make_pool_slice(pool, test_idx)
            model.fit(train_pool)
            scores = {}
            if self._catboost_scoring is not None:
                scores.update(self.eval_model(model, test_pool, metrics=self._catboost_scoring))
            if self._sklearn_scores is not None:
                weights = None
                if self.weight_column is not None:
                    weights = compute_sample_weight('balanced', y=self.weight_column[test_idx])
                scores.update(self._sklearn_scores(model, test_pool, test_pool.get_label(), sample_weight=weights))

            for key in scores:
                scoring_dict[key].append(scores[key])
            if self.optuna_trial is not None:
                if idx == self.n_folds_start_prune:
                    self.optuna_trial.report(np.mean(scoring_dict[self.scoring[0]]), idx)
                    if self.optuna_trial.should_prune():
                        raise TrialPruned()
        return scoring_dict
