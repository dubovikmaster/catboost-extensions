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
from itertools import chain

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
from parallelbar import progress_starmap
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

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
    def _distribute_gpus(available_gpus: List[int], num_folds: int) -> List[List[int]]:
        """
        Distributes the available GPUs across a specified number of folds, ensuring that
        each fold gets at least one GPU. The distribution aims to balance the number
        of GPUs assigned to each fold, iteratively assigning any extra GPUs to the
        earliest folds. In cases where there are more folds than GPUs, some GPUs may
        be reused.

        Parameters
        ----------
        available_gpus : List[int]
            A list of GPU identifiers available for distribution.
        num_folds : int
            The number of folds among which the GPUs need to be distributed.

        Returns
        -------
        List[List[int]]
            A list where each sublist represents the GPUs assigned to a particular fold.

        Notes
        -----
        If the number of GPUs is less than the number of folds, the last GPU in the
        available list will be reused to ensure every fold has at least one GPU.
        Extra GPUs, if present after even distribution, are assigned to the earlier
        folds to balance assignment.
        """
        total_gpus = len(available_gpus)
        base = total_gpus // num_folds
        extra = total_gpus % num_folds
        distribution = []
        current = 0
        for i in range(num_folds):
            num_gpus = base + 1 if i < extra else base
            if num_gpus == 0:
                num_gpus = 1
            assigned_gpus = available_gpus[current:current + num_gpus]
            if not assigned_gpus:
                assigned_gpus = [available_gpus[-1]]
            distribution.append(assigned_gpus)
            current += num_gpus
        return distribution

    @staticmethod
    def _check_cv(cv: Union[int, BaseCrossValidator], model: CatBoostModel) -> BaseCrossValidator:
        if isinstance(cv, int):
            if isinstance(model, CatBoostRegressor):
                _cv = KFold(cv)
            else:
                _cv = StratifiedKFold(cv)
        elif isinstance(cv, BaseCrossValidator):
            return cv
        else:
            raise ValueError('cv must be int or BaseCrossValidator instance')

        return _cv

    @staticmethod
    def get_model_iterations(cb_model: CatBoostModel) -> int:
        iterations = cb_model.get_param('iterations')
        if iterations is None:
            iterations = 1000
        return iterations

    @staticmethod
    def _scoring_prepare(scoring_list):
        scoring_dict = defaultdict(list)
        for score in scoring_list:
            for key in score:
                scoring_dict[key].append(score[key])
        return scoring_dict

    def eval_model(self, cb_model: CatBoostModel, val_pool: Pool, metrics: str) -> dict:
        """
        Evaluates the given CatBoost model on the provided validation data using specified metrics.

        This function computes the evaluation metrics for a CatBoost model on a validation
        dataset, starting computation at the last model iteration. The metrics are returned
        as a dictionary with metric names as keys and their corresponding values.

        Parameters
        ----------
        cb_model : CatBoostModel
            The pre-trained CatBoost model to be evaluated.
        val_pool : Pool
            The validation dataset in CatBoost Pool format, used for evaluating the model.
        metrics : str
            A single evaluation metric or a combination of metrics, as supported by CatBoost.

        Returns
        -------
        dict[str, float]
            Dictionary where keys are the metric names, and values are the corresponding metric values computed from the validation dataset.

        """
        score = cb_model.eval_metrics(val_pool, metrics=metrics, ntree_start=self.get_model_iterations(cb_model) - 1)
        return {key: val[0] for key, val in score.items()}

    def make_pool_slice(self, pool: Pool, idx: ArrayLike) -> Pool:
        """
        Create a sliced pool from the given pool and index.

        This function creates a sliced version of the given pool using the indices
        specified in the idx parameter. If the `weight_column` attribute is set,
        the function computes and assigns sample weights using a balanced strategy
        to the sliced pool. Furthermore, if the `group_id` or `subgroup_id` attributes
        are provided, they will be applied to the resulting sliced pool.

        Parameters
        ----------
        pool : Pool
            The original pool object from which a sliced version will be created.
        idx : ArrayLike
            An array-like object specifying the indices for slicing the pool.

        Returns
        -------
        Pool
            A new pool object representing the sliced version of the original pool,
            potentially updated with weights, group IDs, and subgroup IDs.
        """
        pool_slice = pool.slice(idx)
        if self.weight_column is not None:
            weights = compute_sample_weight('balanced', y=self.weight_column[idx])
            pool_slice.set_weight(weights)
        if self.group_id is not None:
            pool_slice.set_group_id(self.group_id[idx])
        if self.subgroup_id is not None:
            pool_slice.set_subgroup_id(self.subgroup_id[idx])
        return pool_slice

    def _fit_fold(self, pool, train_idx, test_idx, device_ids):
        """
        Fits a fold of the model and evaluates it using specified metrics.

        This method initializes a new copy of the model using the GPU devices specified in `device_ids`.
        It creates training and testing data slices from the given pool and fits the model on the training data.
        The fitted model is then evaluated using both CatBoost metrics and additional specified scikit-learn scoring metrics.
        The method returns the computed evaluation scores as a dictionary.

        Parameters
        ----------
        pool : Pool
            The full data pool from which the train and test subsets are derived.
        train_idx : list of int
            Indices representing the training data in the pool.
        test_idx : list of int
            Indices representing the testing data in the pool.
        device_ids : int or list of int
            GPU device identifier(s) to be used for model fitting.

        Returns
        -------
        dict
            A dictionary containing scores for the evaluation metrics. Scores may include
            those computed from CatBoost metrics and additional scikit-learn metrics if provided.
        """
        model = self.model.copy()
        # Set GPU device
        device_str = ":".join(map(str, device_ids)) if isinstance(device_ids, list) else str(device_ids)
        model.set_params(task_type='GPU', devices=device_str)
        train_pool = self.make_pool_slice(pool, train_idx)
        test_pool = self.make_pool_slice(pool, test_idx)
        model.fit(train_pool)
        scores = {}
        if self._catboost_scoring:
            scores.update(self.eval_model(model, test_pool, metrics=self._catboost_scoring))
        if self._sklearn_scores:
            weights = None
            if self.weight_column is not None:
                weights = compute_sample_weight('balanced', y=self.weight_column[test_idx])
            scores.update(self._sklearn_scores(model, test_pool, test_pool.get_label(), sample_weight=weights))
        return scores

    def _fit_folds(self, pool, trains_idx, tests_idx, device_id):
        """
        Fits multiple folds on the given data split indices.

        This method iterates over the provided train and test indices, applies
        the `_fit_fold` method on each fold, and collects the results. It is
        designed for use in cross-validation or similar tasks where training
        and validation are performed over multiple subsets of data.

        Parameters
        ----------
        pool : Any
            Data pool containing features and labels required for training and
            testing. The specific format of the pool depends on the
            implementation of the `_fit_fold` method.
        trains_idx : list of list of int
            A list containing lists of indices. Each inner list represents the
            indices of the training data for a particular fold.
        tests_idx : list of list of int
            A list containing lists of indices. Each inner list represents the
            indices of the testing data for a particular fold.
        device_id : Any
            Identifier for the computing device (e.g., GPU or CPU) where the
            model training and testing for each fold is executed.

        Returns
        -------
        list
            A list of results returned by the `_fit_fold` method for each fold.
            The structure and type of the results depend on the implementation
            of `_fit_fold`.

        """
        result = list()
        for i in range(len(trains_idx)):
            result.append(self._fit_fold(pool, trains_idx[i], tests_idx[i], device_id))
        return result

    @staticmethod
    def _get_available_gpus() -> List[int]:
        """
            Get the indices of available GPUs on the system.

            This static method checks for available GPUs on the system by querying the NVIDIA System
            Management Interface (nvidia-smi). If it fails to retrieve the GPU information, it defaults
            to assuming a single GPU is present. The indices of the GPUs are returned as a list.

            Returns
            -------
            List[int]
                A list of integers representing the indices of available GPUs. In the case where no GPUs
                are detected or an error occurs, it defaults to a list containing the index 0.
        """
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'])
            num_gpus = len(result.decode('utf-8').strip().split('\n'))
        except Exception:
            num_gpus = 1  # By default, use 1 GPU
        return list(range(num_gpus)) if num_gpus > 0 else [0]

    def parallel_fit(self, show_progress: bool = False, available_gpus: Optional[List[int]] = None) -> dict:
        """
        Fits the model in parallel using cross-validation splits with GPU resources.

        This method performs parallel computation to fit the model across cross-validation
        splits, making use of available GPU resources when possible. If GPU resources are
        not sufficient, it intelligently distributes tasks across available GPUs or falls
        back to other computational cores. The fitting process involves managing data pools,
        splitting data into training and testing subsets for each fold, and distributing
        computational tasks efficiently to maximize resource utilization. Additionally, it
        supports progress display during the computation process.

        Parameters
        ----------
        show_progress : bool, optional (default False)
            If True, displays progress information during the parallel fitting process.
            Defaults to False.
        available_gpus: List[int], optional (default None)
            List of GPU device IDs available for the parallel model.

        Returns
        -------
        List or object
            The aggregated results of the cross-validation fit process, processed by
            an internal scoring preparation method.

        Notes
        -----
        The method ensures that the computational process is distributed optimally
        across available GPUs or CPUs depending on system resource availability. It
        also takes into account the parameter configurations of the model regarding
        text and categorical features as specified during initialization.
        """
        if not isinstance(self.data, Pool):
            pool = Pool(
                self.data,
                self.y,
                text_features=self.model.get_param('text_features'),
                cat_features=self.model.get_param('cat_features'),
            )
        else:
            pool = self.data
        if available_gpus is None:
            available_gpus = self._get_available_gpus()
        splits = self.cv.split(range(pool.shape[0]), self.y)
        n_cpu = min(len(available_gpus), self.cv.n_splits)
        if len(available_gpus) >= self.cv.n_splits:
            gpus_per_fold = self._distribute_gpus(available_gpus, self.cv.n_splits)
            result = progress_starmap(self._fit_fold,
                                      [(pool, train_idx, test_idx, gpus_per_fold[idx]) for
                                       idx, (train_idx, test_idx) in enumerate(splits)], n_cpu=n_cpu,
                                      executor='threads', disable=not show_progress)
        else:
            folds_per_gpu = self._distribute_gpus(list(range(self.cv.n_splits)), len(available_gpus))
            _task = [(train_idx, test_idx) for (train_idx, test_idx) in splits]
            task = list()
            for idx, i in enumerate(folds_per_gpu):
                task.append((pool, [_task[j][0] for j in i], [_task[j][1] for j in i], idx))
            result = list(chain.from_iterable(progress_starmap(self._fit_folds, task, n_cpu=n_cpu, executor='threads',
                                                               disable=not show_progress)))

        return self._scoring_prepare(result)

    def fit(self, show_progress=False) -> dict:
        """
        Fit the model using cross-validation and evaluate scores.

        This method performs training and evaluation of a model using cross-validation. It splits the data
        into training and testing subsets, fits the model on the training subset, and evaluates the model
        on the testing subset. It supports both CatBoost-specific and sklearn scoring mechanisms, and it
        can compute balanced sample weights if provided. Additionally, if an Optuna trial is supplied, it
        reports the score metrics for pruning purposes.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to display progress using a progress bar during cross-validation. Default is False.

        Returns
        -------
        dict
            A dictionary containing scores for each metric as keys. Each value is a list of scores
            obtained from each fold of the cross-validation.
        """
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
        for idx, (train_idx, test_idx) in tqdm(enumerate(splits), disable=not show_progress, total=self.cv.n_splits):
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
