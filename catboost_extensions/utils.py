import logging
import _thread as thread
import threading
import time
from collections import defaultdict
from typing import (
    Callable,
    Optional,
    List,
    Union,
)
from itertools import chain
from contextlib import contextmanager
import warnings

import platform
import signal
import os
import pickle

import pandas as pd
from numpy.typing import ArrayLike
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    KFold,
)
from sklearn import metrics
from sklearn.metrics._scorer import check_scoring
from sklearn.utils.validation import _num_samples

from catboost import (
    Pool,
    CatBoostRanker,
    CatBoostRegressor,
    CatBoostClassifier,
)
from parallelbar import progress_starmap
from tqdm.auto import tqdm

import plotly.express as px

logger = logging.getLogger(__name__)

CatBoostModel = Union[CatBoostRanker, CatBoostRegressor, CatBoostClassifier]
_timer_interrupt = False


def _stop_function():
    """
    Interrupts the main process or thread based on the operating system.

    This function determines the host operating system and performs the appropriate
    procedure to interrupt or stop the current process. On Windows systems, it
    sends an interrupt signal to the main thread. On other systems, it sends a
    SIGINT signal to the current process using its process ID.

    Raises
    ------
    KeyboardInterrupt
        If invoked to stop or interrupt the current process or thread.
    """
    global _timer_interrupt
    _timer_interrupt = True
    if platform.system() == 'Windows':
        thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


@contextmanager
def stop_it_after_timeout(timeout):
    """
    Context manager to stop execution after a specified timeout.

    Parameters
    ----------
    timeout : float
        Maximum time in seconds before raising a TimeoutError.

    Raises
    ------
    TimeoutError
        If the execution exceeds the specified timeout.
    """

    global _timer_interrupt
    _timer_interrupt = False

    timer = threading.Timer(timeout, _stop_function)
    timer.start()

    try:
        yield
    except KeyboardInterrupt:
        if _timer_interrupt:
            raise TimeoutError(f"Execution took more then {timeout} seconds")
        else:
            raise
    finally:
        timer.cancel()


class BootstrapOutOfBag(BaseCrossValidator):
    """
    BootstrapOutOfBag class is a cross-validation generator implementing bootstrap resampling.

    Detailed description of the class, its purpose, and usage.

    Attributes
    ----------
    n_splits : int
        The number of resampling iterations performed during splitting.
    rng : numpy.random.RandomState
        The random number generator used for generating random samples.
    """

    def __init__(self, n_splits=200, random_seed=None):
        """
        Class for managing the split configuration for random state initialization.

        This class initializes a random state generator using a specific random seed
        and sets the number of splits that can be used in further operations or
        experiment configurations.

        Attributes
        ----------
        n_splits : int
            Number of splits to be used; defines how often the data can be split during
            the experimentation or simulation process.
        rng : numpy.random.RandomState
            Random state generator initialized with the provided random seed value.

        Parameters
        ----------
        n_splits : int, optional
            Number of splits to configure. Defaults to 200.
        random_seed : int or None, optional
            Seed value for initializing the random state generator.
            Defaults to None which indicates random initialization.
        """
        self.n_splits = n_splits
        self.rng = np.random.RandomState(random_seed)

    def split(self, X, y=None, groups=None):
        """
        Generates train and test indices for splitting data using a bootstrap resampling method.

        The method splits the data into training and testing sets for a number of splits
        defined by `n_splits`. Each training set is created by sampling with replacement
        from the input data's indices, and the corresponding test set contains the indices
        that are not included in the training set (i.e., the leftover indices).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature data to be split into training and testing sets.

        y : array-like of shape (n_samples,), optional
            The target variable for supervised learning problems. This parameter
            is not used by the method and is included for compatibility with other
            splitting strategies.
        groups : array-like of shape (n_samples,), optional
            This parameter is not used by the method and is included for compatibility with other
            splitting strategies.

        Yields
        ------
        train_idx : ndarray of shape (n_samples_train,)
            The indices of the samples that compose the training set for the current split.

        test_idx : ndarray of shape (n_samples_test,)
            The indices of the samples that compose the testing set for the current split.
        """
        n_samples = _num_samples(X)
        sample_idx = np.arange(n_samples)
        set_idx = set(sample_idx)
        for _ in range(self.n_splits):
            train_idx = self.rng.choice(sample_idx, size=n_samples, replace=True)
            test_idx = np.array(list(set_idx.difference(set(train_idx))))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """ Returns the number of splitting iterations in the cross-validator.

        Parameters
        __________
        X: array-like of shape (n_samples, n_features)
            This parameter is not used by the method and is included for compatibility with other splitting strategies.
        y: array-like of shape (n_samples,), optional
            This parameter is not used by the method and is included for compatibility with other splitting strategies.
        groups: array-like of shape (n_samples,), optional
            This parameter is not used by the method and is included for compatibility with other splitting strategies.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


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
    group_id: Optional[ArrayLike]
        Array of group IDs for multi-group feature settings. Optional.
    subgroup_id: Optional[ArrayLike]
        Array of subgroup IDs for subgroup-specific settings. Optional.
    timeout: Optional[float]
        If timeout is not None and the result does not arrive within timeout seconds then
        multiprocessing.TimeoutError is raised
    """

    def __init__(self, model: CatBoostModel, data: Union[Pool, pd.DataFrame, ArrayLike],
                 y: Optional[Union[pd.Series, pd.DataFrame, ArrayLike]] = None,
                 scoring: Optional[Union[str, List[str], dict[str, Callable]]] = None,
                 cv: Union[BaseCrossValidator, int] = 5,
                 weight_column: Optional[ArrayLike] = None, group_id: Optional[ArrayLike] = None,
                 subgroup_id: Optional[ArrayLike] = None,
                 timeout: Optional[float] = None,
                 save_models: bool = False,
                 ):
        self.model = model
        self.data = data
        self.y = y
        self.scoring = scoring
        self.pool = self._prepare_pool()
        self.cv = cv
        self.weight_column = weight_column
        self.group_id = group_id
        self.subgroup_id = subgroup_id
        self.timeout = timeout
        self.save_models = save_models
        self.models_ = list()
        self.cv_results_ = dict()

    @property
    def scoring(self):
        """
        This property retrieves the value of the `_scoring` attribute.

        The `_scoring` attribute is used to store the scoring configuration or
        method relevant to the use case.

        Returns
        -------
        Any
            The value of the `_scoring` attribute.
        """
        return self._scoring

    @scoring.setter
    def scoring(self, scoring):
        """
        Sets the scoring function and updates internal scoring attributes used for
        evaluating model performance.

        The method ensures that the provided scoring argument is processed
        and converted into appropriate internal attributes representing
        the scoring methods for CatBoost and scikit-learn libraries.

        It automatically updates `_scoring`, `_catboost_scoring`, and
        `_sklearn_scores` attributes to be utilized further in the model code.

        The setter is used to handle changes in the scoring function dynamically,
        allowing proper computation configuration.

        Parameters
        ----------
        scoring : Any
            The scoring function or strategy to be used for evaluating model
            performance. The type and format are dependent on the supported
            scoring methods.

        """
        self._scoring = self._get_score(scoring)
        self._catboost_scoring = self._get_catboost_scores()
        self._sklearn_scores = self._get_sklearn_scores()

    def _get_score(self, scoring):
        """
        Determine the appropriate scoring metric based on the model type.

        This method determines the default scoring metric for a given model if the
        `scoring` parameter is not explicitly provided by the user. Depending on the
        model type, the corresponding metric is selected as the default. If the model
        type is not supported, an exception is raised.

        Parameters
        ----------
        scoring : str or None
            Scoring metric provided by the user. If `None`, a default scoring metric
            will be determined based on the type of the model.

        Returns
        -------
        str
            The scoring metric to be used, either provided by the user or determined as
            the default based on the model type.

        Raises
        ------
        ValueError
            If the model type is not supported.
        """
        if scoring is None:
            if isinstance(self.model, CatBoostRegressor):
                scoring = 'R2'
            elif isinstance(self.model, CatBoostClassifier):
                scoring = 'Accuracy'
            elif isinstance(self.model, CatBoostRanker):
                scoring = 'NDCG'
            else:
                raise ValueError('Model type not supported. Cannot determine default scoring metric.')
            warnings.warn('Setting default scoring metric to: ' + scoring, UserWarning, stacklevel=2)
        return scoring

    def get_n_splits(self):
        """
        Returns the number of splits for a cross-validation strategy.

        The method retrieves the number of splits defined in the cross-validation
        strategy used by the object. This is typically useful for determining how many
        chunks or folds the data will be divided into during cross-validation
        procedures.

        Returns
        -------
        int
            The number of splits in the cross-validation strategy.
        """
        return self.cv.get_n_splits()

    def _get_catboost_scores(self):
        """
        Retrieves applicable CatBoost scoring metrics.

        This method processes the `scoring` attribute of the calling object,
        checking its type and filtering out the scoring metrics that are not
        compatible with CatBoost. The final result is a list of scoring metrics
        suitable for CatBoost.

        Returns
        -------
        list of str
            A list of scoring metric names that are unsupported by CatBoost
            and should be specifically handled or removed in the context of
            CatBoost scoring.
        """
        scoring = self.scoring
        if isinstance(scoring, str):
            scoring = [scoring]
        if not isinstance(scoring, dict):
            return [i for i in scoring if i not in metrics.get_scorer_names()]

    def _get_sklearn_scores(self):
        """
        _get_sklearn_scores(self)

        Determines and retrieves scikit-learn compatible scoring methods based on the
        `scoring` attribute. The function processes the `scoring` attribute, which can
        be a string, list, or dictionary, and validates its compatibility with
        scikit-learn scoring standards. If compatible scoring measures are identified,
        the corresponding scikit-learn scoring objects are retrieved.

        Returns
        -------
        callable or list of callable
            A scoring callable or a list of scoring callables compatible with
            scikit-learn, depending on the format of the input `scoring`. The callable(s)
            can be used for model evaluation based on the specified scoring criteria.

        Parameters
        ----------
        None

        Raises
        ------
        None
        """
        scoring = self.scoring
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

    def _check_cv(self, cv: Union[int, BaseCrossValidator]) -> BaseCrossValidator:
        """
        Check the cross-validator object and ensure it is properly initialized.

        This function verifies if the provided cross-validator is an integer or an
        instance of a `BaseCrossValidator`. If the input is an integer, it initializes
        a `KFold` object for regression models or a `StratifiedKFold` object for
        classification models. If the input is already an instance of
        `BaseCrossValidator`, it is returned as-is. If the input does not match these
        criteria, a `ValueError` is raised.

        Parameters
        ----------
        cv : int or BaseCrossValidator
            Specifies the cross-validation splitting strategy. If an integer is
            provided, it determines the number of splits, and a default cross-validator
            is initialized.

        Returns
        -------
        BaseCrossValidator
            The valid cross-validator object, either provided or newly initialized.
        """
        if isinstance(cv, int):
            if isinstance(self.model, CatBoostRegressor):
                _cv = KFold(cv)
            else:
                _cv = StratifiedKFold(cv)
        elif isinstance(cv, BaseCrossValidator):
            return cv
        else:
            raise ValueError('cv must be int or BaseCrossValidator instance')

        return _cv

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, cv):
        self._cv = self._check_cv(cv)

    @staticmethod
    def get_model_iterations(cb_model: CatBoostModel) -> int:
        """
        Determines the number of iterations used by a CatBoost model.

        This function fetches the value of the 'iterations' parameter from a
        CatBoost model object. If the parameter is not explicitly set, it will
        default to a value of 1000 and return this value.

        Parameters
        ----------
        cb_model : CatBoostModel
            The CatBoost model from which the 'iterations' parameter needs
            to be fetched.

        Returns
        -------
        int
            The number of iterations set in the model or the default value
            of 1000 if not explicitly set.
        """
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
            Dictionary where keys are the metric names, and values are the corresponding metric values computed from
            the validation dataset.

        """
        score = cb_model.eval_metrics(val_pool, metrics=metrics, ntree_start=self.get_model_iterations(cb_model) - 1)
        return {key: val[0] for key, val in score.items()}

    def make_pool_slice(self, idx: ArrayLike) -> Pool:
        """
        Create a sliced pool from the given pool and index.

        This function creates a sliced version of the given pool using the indices
        specified in the idx parameter. If the `weight_column` attribute is set,
        the function computes and assigns sample weights using a balanced strategy
        to the sliced pool. Furthermore, if the `group_id` or `subgroup_id` attributes
        are provided, they will be applied to the resulting sliced pool.

        Parameters
        ----------
        idx : ArrayLike
            An array-like object specifying the indices for slicing the pool.

        Returns
        -------
        Pool
            A new pool object representing the sliced version of the original pool,
            potentially updated with weights, group IDs, and subgroup IDs.
        """
        pool_slice = self.pool.slice(idx)
        if self.weight_column is not None:
            weights = compute_sample_weight('balanced', y=self.weight_column[idx])
            pool_slice.set_weight(weights)
        if self.group_id is not None:
            pool_slice.set_group_id(self.group_id[idx])
        if self.subgroup_id is not None:
            pool_slice.set_subgroup_id(self.subgroup_id[idx])
        return pool_slice

    def _prepare_pool(self):
        """
        _prepare_pool(self)

        Prepares a Pool object for use with machine learning models. If the `self.data`
        attribute is already an instance of the Pool class, it is returned as-is.
        Otherwise, a new Pool is constructed using the `self.data` and `self.y`
        attributes, along with text and categorical feature parameters retrieved
        from the model.

        Returns
        -------
        Pool
            The prepared Pool object containing the input data, labels, and feature
            configurations, suitable for use in model training or evaluation.

        Parameters
        ----------
        self : object
            The instance of the class where this method is defined and executed. It
            must contain the attributes `data`, `y`, and `model` to operate correctly.
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
        return pool

    def _fit_fold(self, train_idx, test_idx, device_ids=None):
        """
        Fits a fold of the model and evaluates it using specified metrics.

        This method initializes a new copy of the model using the GPU devices specified in `device_ids`.
        It creates training and testing data slices from the given pool and fits the model on the training data.
        The fitted model is then evaluated using both CatBoost metrics and additional specified scikit-learn scoring
        metrics. The method returns the computed evaluation scores as a dictionary.

        Parameters
        ----------
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
        if device_ids is not None:
            device_str = ":".join(map(str, device_ids)) if isinstance(device_ids, list) else str(device_ids)
            model.set_params(task_type='GPU', devices=device_str)
        train_pool = self.make_pool_slice(train_idx)
        test_pool = self.make_pool_slice(test_idx)
        start = time.perf_counter()
        model.fit(train_pool)
        fit_time = round(time.perf_counter() - start, 2)
        scores = {}
        if self._catboost_scoring:
            scores.update(self.eval_model(model, test_pool, metrics=self._catboost_scoring))
        if self._sklearn_scores:
            weights = None
            if self.weight_column is not None:
                weights = compute_sample_weight('balanced', y=self.weight_column[test_idx])
            scores.update(self._sklearn_scores(model, test_pool, test_pool.get_label(), sample_weight=weights))
        scores['fit_time'] = fit_time
        if self.save_models:
            self.models_.append(model)
        for key, values in scores.items():
            self.cv_results_[key].append(values)
        return scores

    def _fit_folds(self, trains_idx, tests_idx, device_id):
        """
        Fits multiple folds on the given data split indices.

        This method iterates over the provided train and test indices, applies
        the `_fit_fold` method on each fold, and collects the results. It is
        designed for use in cross-validation or similar tasks where training
        and validation are performed over multiple subsets of data.

        Parameters
        ----------
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
            result.append(self._fit_fold(trains_idx[i], tests_idx[i], device_id))
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
        Raises
        ______
            TimeoutError: If the fitting process exceeds the specified timeout duration.

        Notes
        -----
        The method ensures that the computational process is distributed optimally
        across available GPUs or CPUs depending on system resource availability. It
        also takes into account the parameter configurations of the model regarding
        text and categorical features as specified during initialization.
        """
        self.models_ = list()
        self.cv_results_ = defaultdict(list)
        if available_gpus is None:
            available_gpus = self._get_available_gpus()
        splits = self.cv.split(range(self.pool.shape[0]), self.y)
        n_splits = self.get_n_splits()
        n_cpu = min(len(available_gpus), n_splits)
        if len(available_gpus) >= n_splits:
            gpus_per_fold = self._distribute_gpus(available_gpus, n_splits)
            result = progress_starmap(self._fit_fold,
                                      [(train_idx, test_idx, gpus_per_fold[idx]) for
                                       idx, (train_idx, test_idx) in enumerate(splits)], n_cpu=n_cpu,
                                      executor='threads', disable=not show_progress, timeout=self.timeout)
        else:
            folds_per_gpu = self._distribute_gpus(list(range(n_splits)), len(available_gpus))
            _task = [(train_idx, test_idx) for (train_idx, test_idx) in splits]
            task = list()
            for idx, i in enumerate(folds_per_gpu):
                task.append(([_task[j][0] for j in i], [_task[j][1] for j in i], idx))
            result = list(chain.from_iterable(progress_starmap(self._fit_folds, task, n_cpu=n_cpu, executor='threads',
                                                               disable=not show_progress, timeout=self.timeout)))

        return self._scoring_prepare(result)

    def _fit(self, show_progress=False) -> dict:
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
        splits = self.cv.split(range(self.pool.shape[0]), self.y)
        scoring_dict = defaultdict(list)
        for (train_idx, test_idx) in tqdm(splits, disable=not show_progress, total=self.get_n_splits()):
            scores = self._fit_fold(train_idx, test_idx)
            for key in scores:
                scoring_dict[key].append(scores[key])
        return scoring_dict

    def fit(self, show_progress=False):
        """
        Fits the model using the provided data and settings within a specified timeout duration.

        This method attempts to execute the fitting process for the model within a given
        time frame. A separate thread-based timer is used to monitor the time limit for
        the cross-validation process. If the time exceeds the predefined limit, a custom
        TimeoutError is raised. The method also handles interruptions triggered by the
        user, ensuring proper cleanup of resources.

       Parameters
        ----------
        show_progress : bool, optional
            Whether to display progress using a progress bar during cross-validation. Default is False.

        Returns
        -------
        dict
            A dictionary containing scores for each metric as keys. Each value is a list of scores
            obtained from each fold of the cross-validation.
        Raises
        ______
            TimeoutError: If the fitting process exceeds the specified timeout duration.
        """
        self.models_ = list()
        self.cv_results_ = defaultdict(list)
        with stop_it_after_timeout(self.timeout):
            result = self._fit(show_progress)
        return result

    def ifit(self, timeout=None):
        """
        Executes an iterative fit method with optional timeout handling, yielding performance
        scores for each cross-validation fold.

        This method divides the provided dataset into training and testing indices using the
        defined cross-validation strategy and fits the model iteratively for each fold. It
        handles timeout settings, splitting time proportionally across the folds, to ensure
        timely processing. Yields performance scores for the model on each fold during the
        execution.

        Parameters
        ----------
        timeout : float, optional
            The maximum amount of time allowed (in seconds) for the fitting process across
            all cross-validation folds. If not specified, defaults to the instance's timeout
            value. If the instance's timeout is also not set, no timeout is applied. If a
            timeout is applied, the allocated time will be divided equally across the folds.

        Yields
        ------
        scores : Any
            The performance scores of the model on the testing set for each fold, as
            determined by the `_fit_fold` function.
        """
        self.models_ = list()
        self.cv_results_ = defaultdict(list)
        if timeout is None:
            timeout = self.timeout
            if timeout is not None:
                timeout /= self.get_n_splits()
        splits = self.cv.split(range(self.pool.shape[0]), self.y)
        for (train_idx, test_idx) in splits:
            with stop_it_after_timeout(timeout):
                scores = self._fit_fold(train_idx, test_idx)
            yield scores

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'pool' in state:
            del state['pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.pool = self._prepare_pool()

    def save(self, file_name):
        """
        Saves the current object instance to a file using pickle.

        This method serializes the instance and stores it as a `.pkl` file with the
        specified filename. The method ensures the proper handling of file writing
        by using a context manager. The file will be created in the current working
        directory with the specified filename appended with a `.pkl` extension.

        Parameters
        ----------
        file_name : str
            The base name (without extension) for the file to which the object
            will be saved. The method appends a `.pkl` extension to this name
            during the saving process.

        """
        with open(f'{file_name}.pkl', 'wb') as f:  # open a text file
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_name):
        """
        Loads a serialized object from a file.

        The method loads an object previously serialized and stored in a file with
        pickle library. It expects the input file to be in the ".pkl" format. The method
        reads the file in binary mode, deserializes the stored object, and returns it.

        Parameters
        ----------
        file_name : str
            The name of the file (without the extension '.pkl') from which the object
            will be loaded.

        Returns
        -------
        obj : object
            The deserialized object loaded from the specified file.
        """
        with open(f'{file_name}.pkl', 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def _add_percentile_to_figure(fig, score, line_color, horizontal_line=True):
        lower = np.percentile(score, 2.5)
        upper = np.percentile(score, 97.5)
        if horizontal_line:
            fig.add_hline(y=lower, line_dash="dash", annotation_text='2.5 Pctl', line_color=line_color)
            fig.add_hline(y=upper, line_dash="dash", annotation_text='97.5 Pctl', line_color=line_color)
        else:
            fig.add_vline(x=lower, line_dash="dash", annotation_text='2.5 Pctl', line_color=line_color)
            fig.add_vline(x=upper, line_dash="dash", annotation_text='97.5 Pctl', line_color=line_color)
        return fig

    def plot_score(self, score: str, compare_with: Optional[ArrayLike] = None, log_scale: bool = False,
                   plot_type: str = 'box', height: Optional[int] = None, width: Optional[int] = None,
                   show_confidence_interval=False, **kwargs
                   ):
        """
        Generates and returns a plot figure for the specified scoring metric based on the cross-validation results.
        The figure can be a boxplot or line plot showcasing the distribution or trends of the scores.

        Parameters
        ----------
        score : str
            The name of the score metric to visualize. It must be a key present in the cross-validation results.
        compare_with: ArrayLike, default None
        log_scale : bool, optional
            Indicates whether the Y-axis of the plot should use a logarithmic scale. Default is False.

        plot_type : str, optional
            The type of plot to generate. Acceptable values are:
            - 'box': Generates a boxplot for the score.
            - 'line': Generates a line plot for the score.
            Default is 'box'.

        height : int, optional
            The height of the plot in pixels. Default value is None, which uses the plotting library's default.

        width : int, optional
            The width of the plot in pixels. Default value is None, which uses the plotting library's default.
        kwargs: dict
        show_confidence_interval: bool, default False
        Raises
        ------
        ValueError
            If cross-validation results are not available, or if the specified score is not found in the results, or
            if an unsupported plot_type is provided.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object representing the specified visualization of the given score.
        """
        if not self.cv_results_:
            raise ValueError('You must run one of "fit", "parallel_fit" or "ifit" first')
        if score not in self.cv_results_:
            raise ValueError('Score not found')
        df = pd.DataFrame(self.cv_results_)
        df['fold'] = list(range(self.get_n_splits()))
        color = None
        if compare_with is not None:
            df_compare = pd.DataFrame({score: compare_with})
            df_compare['fold'] = list(range(len(df_compare)))
            df_compare['group'] = 'B'
            df['group'] = 'A'
            df = pd.concat([df, df_compare])
            color = 'group'
        if plot_type == 'box':
            fig = px.box(
                df,
                points="all",
                title=f'Boxplot for {score}',
                y=score,
                hover_data=['fold'],
                log_y=log_scale,
                color=color,
                height=height,
                width=width,
                **kwargs,
            )
        elif plot_type == 'line':
            fig = px.line(
                df,
                x='fold',
                y=score,
                title=f'Line plot for {score}',
                markers=True,
                height=height,
                width=width,
                color=color,
                **kwargs,
            )
        elif plot_type == 'hist':
            fig = px.histogram(
                df,
                x=score,
                title=f'Histogram for {score}',
                text_auto=True,
                marginal="rug",
                height=height,
                width=width,
                color=color,
                **kwargs,
            )
        else:
            ValueError('Got unexpected plot type. Should be "box" or "line"')
        if show_confidence_interval:
            horizontal_line = True
            if plot_type=='hist':
                horizontal_line = False
            # Confidence interval
            if compare_with is not None:
                fig = self._add_percentile_to_figure(fig, df.loc[df['group'] == 'A', score], 'green',
                                                     horizontal_line = horizontal_line)
                fig = self._add_percentile_to_figure(fig, df.loc[df['group'] == 'B', score], 'black',
                                                     horizontal_line=horizontal_line)
            else:
                fig = self._add_percentile_to_figure(fig, df[score], 'green', horizontal_line=horizontal_line)
        return fig
