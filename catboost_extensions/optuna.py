from typing import (
    Callable,
    Optional,
    Union,
    List,
    Dict,
    Tuple,
)
import logging
import warnings
import pprint

import numpy as np
from numpy.typing import ArrayLike

from optuna.distributions import (
    BaseDistribution,
    IntDistribution,
    FloatDistribution,
    CategoricalDistribution
)
from optuna.trial import Trial
from optuna.exceptions import TrialPruned

import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator
)

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    CatBoostRanker,
)

from .utils import (
    stopit_after_timeout,
    CrossValidator
)

logger = logging.getLogger('optuna.OptunaTuneCV')

CatboostModel = Union[CatBoostRegressor, CatBoostClassifier, CatBoostRanker]
DataSet = Union[np.ndarray, List, pd.DataFrame, pd.Series]


class HyperParam:
    _ATTRS_FOR_TYPE = {
        'int': ('low', 'high', 'step', 'log'),
        'float': ('low', 'high', 'log'),
        'categorical': ('choices',),
    }

    def __init__(self, default_dist):
        """
        Represents a basic initialization class for managing a default distribution value.

        The class primarily initializes the instance with a provided default distribution
        value and sets an additional attribute to None by default. This structure can act
        as a foundation for further properties or methods to work with the given
        distribution data.
        Parameters
        __________
        default_dist: The default value for the distribution to be stored during  initialization.
        """
        self.default_dist = default_dist
        self.attr_name = None

    def __set_name__(self, owner, name):
        self.attr_name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.attr_name, None)

    def __set__(self, instance, update: Union[dict, list, tuple, BaseDistribution]):
        """
        Updates an instance's attribute with a new distribution configuration.

        This descriptor method manages the update of an instance's distribution attribute. It determines how to
        configure a new distribution based on the provided `update` input, ensuring proper compatibility with the
        existing distribution type. If the distribution type changes, a new distribution object is created. When the
        type remains the same, the update is applied in-place. The method guarantees consistency between the updated
        distribution and the associated parameters within the instance's internal state.

        Parameters
        __________
        instance: The instance whose attribute needs to be updated. It must contain an `_params` dictionary
            attribute used for internal parameter synchronization.
        update: A `dict`, `list`, `tuple` or BaseDistribution instance specifying the update details for the distribution. For a
            `list` or `tuple`, it may represent categorical choices or a pair of minimum and maximum values. If a
            `dict` is provided, it should contain the new configuration or changes, including optional `dist_type`.
        Returns: None.
            The instance's attribute and internal parameters are updated in-place.
        """
        if isinstance(update, BaseDistribution):
            dist = update
        else:
            current_dist = getattr(instance, self.attr_name, None)
            if current_dist is None:
                current_dist = self.default_dist
            old_type = self._get_dist_type_str(current_dist)
            if isinstance(update, (list, tuple)):
                if old_type == 'categorical':
                    update = {'choices': update}
                else:
                    if len(update) != 2:
                        raise ValueError('You must specify a minimum and maximum value to create a distribution')
                    low, high = sorted(update)
                    update = {'low': low, 'high': high}

            new_type = update.get('dist_type', old_type)
            kwargs = {key: val for key, val in update.items() if key != 'dist_type'}
            if new_type != old_type:
                dist = self._create_distribution_from_kwargs(new_type, **kwargs)
            else:
                dist = self._update_in_place(current_dist, old_type, kwargs)

        setattr(instance, self.attr_name, dist)
        instance._params[self.attr_name[1:]] = dist

    def _update_in_place(self, dist, dist_type: str, updates: dict):
        """
        Updates the attributes of a given distribution object in place, ensuring
        that only allowed attributes for the specified distribution type are
        updated. This method modifies the object directly without creating a new
        instance.

        Parameters
        __________
        dist: The distribution object whose attributes are to be updated.
        dist_type: The type of the distribution, used to validate allowed
            attributes.
        updates: A dictionary containing the attributes to update and their
            corresponding new values. Keys in the dictionary must match allowed
            attributes for the given distribution type.
        Returns: The updated distribution object with modifications applied.
        """
        allowed_attrs = self._ATTRS_FOR_TYPE[dist_type]

        for key, val in updates.items():
            if key not in allowed_attrs:
                raise ValueError(
                    f"Attribute '{key}' not allowed for distribution type '{dist_type}'"
                )
            setattr(dist, key, val)

        return dist

    @classmethod
    def _create_distribution_from_kwargs(cls, dist_type: str, **kwargs):
        """
        Create a new distribution instance based on the provided distribution type and
        corresponding keyword arguments.

        This method parses the `dist_type` to identify the appropriate distribution
        class for instantiation. Valid distribution types include 'int', 'float',
        and 'categorical'. Depending on the `dist_type`, it constructs and returns
        an instance of the corresponding distribution with the provided arguments.

        Parameters
        __________
        dist_type: A string specifying the type of distribution to create.
                          Supported values are 'int', 'float', and 'categorical'.
        kwargs: Arbitrary keyword arguments required to initialize the chosen
                       distribution type.
        Returns: An instance of `IntDistribution`, `FloatDistribution`, or
                 `CategoricalDistribution`, depending on `dist_type`.
        Raises: ValueError: If the provided `dist_type` is not recognized or supported.
        """
        if dist_type == 'int':
            return IntDistribution(**kwargs)
        elif dist_type == 'float':
            return FloatDistribution(**kwargs)
        elif dist_type == 'categorical':
            return CategoricalDistribution(**kwargs)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    @classmethod
    def _get_dist_type_str(cls, dist) -> str:
        """
        Determines and returns the type of a given distribution as a string.

        This method evaluates the type of the provided distribution object and maps
        it to a corresponding string representation. The supported types include
        integer, float, and categorical distributions. If an unsupported distribution
        type is passed, an exception will be raised.

        :param dist: The distribution to be evaluated.
        :type dist: Union[IntDistribution, FloatDistribution, CategoricalDistribution]
        :return: A string representing the type of the distribution.
        :rtype: str
        :raises ValueError: If the `dist` is not of a supported distribution type.
        """
        if isinstance(dist, IntDistribution):
            return 'int'
        elif isinstance(dist, FloatDistribution):
            return 'float'
        elif isinstance(dist, CategoricalDistribution):
            return 'categorical'
        else:
            raise ValueError(f"Unsupported distribution type: {dist}")


def update_params(fn):
    """Decorator for updating _param dict in CatboostParamSpace class"""

    def wrapper(self, value):
        result = fn(self, value)
        param_name = fn.__name__
        self._params[param_name] = getattr(self, param_name)
        return result

    return wrapper


class CatboostParamSpace:
    iterations = HyperParam(IntDistribution(100, 5000))
    learning_rate = HyperParam(FloatDistribution(1e-3, 1e-1, log=True))
    depth = HyperParam(IntDistribution(2, 15))
    grow_policy = HyperParam(CategoricalDistribution(['SymmetricTree', 'Depthwise', 'Lossguide']))
    l2_leaf_reg = HyperParam(FloatDistribution(1e-2, 10000.0, log=True))
    random_strength = HyperParam(FloatDistribution(1e-2, 10.0, log=True))
    bootstrap_type = HyperParam(CategoricalDistribution(['Bayesian', 'MVS', 'Bernoulli', 'No']))
    score_function = HyperParam(CategoricalDistribution(['Cosine', 'L2']))
    rsm = HyperParam(FloatDistribution(0.01, 1.0))
    simple_ctr_type = HyperParam(CategoricalDistribution(
        ['Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter']))
    combinations_ctr_type = HyperParam(CategoricalDistribution(
        ['Borders', 'Buckets', 'BinarizedTargetMeanValue', 'Counter']))
    simple_ctr_border_type = HyperParam(CategoricalDistribution(['Uniform']))
    combinations_ctr_border_type = HyperParam(CategoricalDistribution(['Uniform']))
    target_border_type = HyperParam(CategoricalDistribution(
        ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy', 'GreedyLogSum']))
    max_ctr_complexity = HyperParam(IntDistribution(1, 10))
    max_bin = HyperParam(IntDistribution(8, 512))
    min_data_in_leaf = HyperParam(IntDistribution(1, 32))
    one_hot_max_size = HyperParam(IntDistribution(2, 255))
    bagging_temperature = HyperParam(FloatDistribution(0, 10))
    subsample = HyperParam(FloatDistribution(0.1, 1, log=True))
    simple_ctr_border_count = HyperParam(IntDistribution(1, 255))
    combinations_ctr_border_count = HyperParam(IntDistribution(1, 255))
    leaf_estimation_method = HyperParam(CategoricalDistribution(['Newton', 'Gradient']))
    boost_from_average = HyperParam(CategoricalDistribution([True, False]))
    model_size_reg = HyperParam(FloatDistribution(1e-2, 10, log=True))
    langevin = HyperParam(CategoricalDistribution([True, False]))
    boosting_type = HyperParam(CategoricalDistribution(['Ordered', 'Plain']))
    fold_len_multiplier = HyperParam(FloatDistribution(1.1, 2))

    def __init__(self, task_type: str = 'CPU', cook_params: Optional[list] = None, params_preset: str = 'general'):
        self.cook_params = cook_params
        self.params_preset = params_preset
        self._params = {}
        self._task_type = task_type
        self.iterations = {}
        self.learning_rate = {}
        self.depth = {}
        self.grow_policy = {}
        self.l2_leaf_reg = {}
        self.random_strength = {}
        self.bootstrap_type = {}
        self.score_function = {}
        self.rsm = {}
        self.simple_ctr_type = {}
        self.combinations_ctr_type = {}
        self.simple_ctr_border_type = {}
        self.combinations_ctr_border_type = {}
        self.target_border_type = {}
        if self._task_type == 'GPU':
            self.bootstrap_type = ['Bayesian', 'Poisson', 'Bernoulli', 'MVS', 'No']
            self.score_function = ['Cosine', 'L2', 'NewtonCosine', 'NewtonL2']
            self.simple_ctr_type = ['Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue']
            self.combinations_ctr_type = ['Borders', 'Buckets', 'FeatureFreq', 'FloatTargetMeanValue']
            self.simple_ctr_border_type = ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum', 'MinEntropy',
                                           'GreedyLogSum']
            self.combinations_ctr_border_type = ['Median', 'Uniform']
        self.max_ctr_complexity = {}
        self.max_bin = {}
        self.min_data_in_leaf = {}
        self.one_hot_max_size = {}
        self.bagging_temperature = {}
        self.subsample = {}
        self.simple_ctr_border_count = {}
        self.combinations_ctr_border_count = {}
        self.leaf_estimation_method = {}
        self.boost_from_average = {}
        self.model_size_reg = {}
        self.langevin = {}
        self.boosting_type = {}
        self.fold_len_multiplier = {}
        self._params = self._get_params_presets()

    def __str__(self):
        return pprint.pformat(self._params, indent=2)

    @property
    def task_type(self):
        return self._task_type

    @task_type.setter
    @update_params
    def task_type(self, value):
        if value not in ['CPU', 'GPU']:
            raise ValueError('task_type must be "CPU" or "GPU"')
        self._task_type = value

    def _get_params_presets(self):
        """ Get parameters presets"""

        if self.cook_params is not None:
            params = {
                name: getattr(self, name)
                for name in self.cook_params
            }
        elif self.params_preset == 'small':
            params = {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'grow_policy': self.grow_policy,
                'l2_leaf_reg': self.l2_leaf_reg,
                'random_strength': self.random_strength,
                'bootstrap_type': self.bootstrap_type,
            }

        elif self.params_preset == 'general':
            params = {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'grow_policy': self.grow_policy,
                'l2_leaf_reg': self.l2_leaf_reg,
                'random_strength': self.random_strength,
                'bootstrap_type': self.bootstrap_type,
                'max_bin': self.max_bin,
                'score_function': self.score_function,
            }
            if self.task_type == 'CPU':
                params['rsm'] = self.rsm
        elif self.params_preset == 'extended':
            params = {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'grow_policy': self.grow_policy,
                'l2_leaf_reg': self.l2_leaf_reg,
                'random_strength': self.random_strength,
                'bootstrap_type': self.bootstrap_type,
                'max_bin': self.max_bin,
                'min_data_in_leaf': self.min_data_in_leaf,
                'score_function': self.score_function,
                'leaf_estimation_method': self.leaf_estimation_method,
                'boost_from_average': self.boost_from_average,
            }
            if self.task_type == 'CPU':
                params['rsm'] = self.rsm
        elif self.params_preset == 'ctr':
            params = {
                'one_hot_max_size': self.one_hot_max_size,
                'max_ctr_complexity': self.max_ctr_complexity,
                'model_size_reg': self.model_size_reg,
                'simple_ctr': f'{self.simple_ctr_type}:CtrBorderCount={self.simple_ctr_border_count}:' \
                              f'CtrBorderType={self.simple_ctr_border_type}',
                'combinations_ctr': f'{self.combinations_ctr_type}:CtrBorderCount={self.combinations_ctr_border_count}:' \
                                    f'CtrBorderType={self.combinations_ctr_border_type}'
            }
        else:
            raise ValueError('params_type must be "extended", "general", "ctr" or "small"')

        return params

    def add_params(self, params: List[str]):
        """ Add parameters to the parameter space
        Parameters
        ----------
        params : List[str]
            List of parameters to add
        """
        for param in params:
            self._params[param] = getattr(self, param)

    def del_params(self, params: List[str]):
        """ Delete parameters from the parameter space
        Parameters
        ----------
        params : List[str]
            List of parameters to delete
        """
        for param in params:
            self._params.pop(param)

    def get_params_space(self):
        """ Show parameters space"""
        return self._params

    @staticmethod
    def get_ctr_params(
            ctr_type='Borders',
            ctr_border_count=15,
            ctr_border_type='Uniform',
            target_border_count=1,
            target_border_type='MinEntropy',
    ):
        """ Get CTR parameters"""
        return f'{ctr_type}:CtrBorderCount={ctr_border_count}:CtrBorderType={ctr_border_type}'

    def __call__(self, trial):

        if self.params_preset in ['small', 'general', 'extended'] or self.cook_params is not None:
            params = {
                name: trial._suggest(name, distribution)
                for name, distribution in self._params.items()
            }
            if self.params_preset in ['small', 'general', 'extended']:
                if params['bootstrap_type'] != 'No':
                    if params["bootstrap_type"] == "Bayesian":
                        params["bagging_temperature"] = trial._suggest("bagging_temperature", self.bagging_temperature)
                    else:
                        params["subsample"] = trial._suggest("subsample", self.subsample)
        elif self.params_preset == 'ctr':
            params = {
                'one_hot_max_size': trial._suggest('one_hot_max_size', self.one_hot_max_size),
                'max_ctr_complexity': trial._suggest('max_ctr_complexity', self.max_ctr_complexity),
                'model_size_reg': trial._suggest('model_size_reg', self.model_size_reg),
            }
            simple_ctr_type = trial._suggest('simple_ctr_type', self.simple_ctr_type)
            combinations_ctr_type = trial._suggest('combinations_ctr_type', self.combinations_ctr_type)
            simple_ctr_border_count = trial._suggest('simple_ctr_border_count', self.simple_ctr_border_count)
            combinations_ctr_border_count = trial._suggest('combinations_ctr_border_count',
                                                           self.combinations_ctr_border_count)
            if simple_ctr_type != 'FeatureFreq':
                if self.task_type == 'GPU':
                    simple_ctr_border_type = trial._suggest('simple_ctr_border_type',
                                                            self.simple_ctr_border_type)
                else:
                    simple_ctr_border_type = 'Uniform'

            else:
                simple_ctr_border_type = 'MinEntropy'

            if combinations_ctr_type != 'FeatureFreq':
                if self.task_type == 'GPU':
                    combinations_ctr_border_type = trial._suggest('combinations_ctr_border_type',
                                                                  self.combinations_ctr_border_type)
                else:
                    combinations_ctr_border_type = 'Uniform'
            else:
                combinations_ctr_border_type = 'Median'
            params['simple_ctr'] = self.get_ctr_params(
                ctr_type=simple_ctr_type,
                ctr_border_count=simple_ctr_border_count,
                ctr_border_type=simple_ctr_border_type
            )
            params['combinations_ctr'] = self.get_ctr_params(
                ctr_type=combinations_ctr_type,
                ctr_border_count=combinations_ctr_border_count,
                ctr_border_type=combinations_ctr_border_type
            )
        else:
            raise ValueError('params_preset must be "extended", "general", "ctr" or "small"')
        return params


class OptunaTuneCV:
    """
    Class for optimizing model hyperparameters using Optuna with cross-validation.

    This class acts as a callable objective function for Optuna studies. It automates
    hyperparameter tuning for machine learning models via Optuna, integrating with
    cross-validation to evaluate performance. It supports parameter space definitions,
    scoring customization, and optional multi-GPU parallelism.

    Attributes
    ----------
        model : CatboostModel
            The model whose hyperparameters are being tuned.
        param_distributions : Union[Dict, Callable[[Trial], Dict]]
            Hyperparameter search space, either as a dictionary or callable returning
            parameters per trial.
        direction : str
            Optimization direction, either 'maximize' or 'minimize'.
        x : DataSet
            The feature data for training.
        y : DataSet
            The target variable corresponding to the training data.
        group_id : Optional[List[int]]
            Optional group identifiers for the data.
        cv : Union[int, BaseCrossValidator]
            Cross-validation splitting strategy, e.g., number of folds or custom validator.
        scoring : Optional[str]
            Scoring metric to evaluate model performance.
        params_post_processing : Optional[Callable[[Trial, Dict], Dict]]
            Callable for post-processing hyperparameters after they are sampled.
        _best_score : float
            Internal tracking of the best score seen so far.
        best_score : Optional[float]
            External tracking for the best score, if provided at initialization.
        weight_column : Optional[ArrayLike]
            Optional sample weight for training data.
        n_folds_start_prune : int
            Minimum number of completed folds before pruning can activate. Default is infinity.
        has_pruner : bool
            Whether pruning logic is enabled for optimization.
        parallel : bool
            Whether to enable parallel cross-validation runs.
        parallel_available_gpus : Optional[List[int]]
            List of available GPU IDs for parallel computation, if applicable.
        trial_timeout : Optional[int]
            Maximum duration (in seconds) per trial before pruning. Ignored if `parallel` is True.
        error_handling : str
            Strategy for handling exceptions during trials. Options are 'raise' or 'prune'.
    Examples
    --------
    >>> from catboost import CatBoostClassifier
    >>> from catboost_extensions import OptunaTuneCV, CatboostParamSpace
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from sklearn.metrics import roc_auc_score
    >>> import optuna

    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=0)
    >>> model = CatBoostClassifier(task_type='CPU', verbose=0)
    >>> param_space_getter = CatboostParamSpace(params_preset='general', task_type='GPU')
    >>> objective = OptunaTuneCV(model, param_space_getter, X, y, cv=5, scoring='roc_auc', direction='maximize')
    >>> study = optuna.create_study(direction='maximize')
    >>> study.optimize(objective, n_trials=100)

    """

    def __init__(
            self,
            model: CatboostModel,
            param_space: Union[Dict, Callable[[Trial], Dict]],
            x: DataSet,
            y: DataSet,
            group_id: Optional[List[int]] = None,
            last_best_score: Optional[float] = None,
            trial_timeout: Optional[int] = None,
            params_post_processing: Optional[Callable[[Trial, Dict], Dict]] = None,
            cv: Union[int, BaseCrossValidator] = 5,
            scoring: Optional[str] = None,
            direction: str = 'maximize',
            weight_column: Optional[ArrayLike] = None,
            has_pruner: bool = False,
            n_folds_start_prune: int = np.inf,
            parallel: bool = False,
            parallel_available_gpus: Optional[List[int]] = None,
            error_handling: str = 'raise',
    ):
        self.model = model
        self.param_distributions = param_space
        self.direction = direction
        self.x = x
        self.y = y
        self.group_id = group_id
        self.cv = cv
        self.scoring = scoring
        self.params_post_processing = params_post_processing
        self._best_score = -np.inf if direction == 'maximize' else np.inf
        self.best_score = last_best_score
        self.weight_column = weight_column
        self.n_folds_start_prune = n_folds_start_prune
        self.has_pruner = has_pruner
        self.parallel = parallel
        self.parallel_available_gpus = parallel_available_gpus
        if self.has_pruner:
            warnings.warn(
                "The 'has_pruner' argument is deprecated and will be removed in a future version. ",
                DeprecationWarning,
                stacklevel=2
            )
        self.trial_timeout = trial_timeout
        if self.trial_timeout is not None:
            if self.parallel:
                warnings.warn('Trial timeout is not supported when parallel mode is enabled. Ignoring this parameter.')
        self.error_handling = error_handling

    @property
    def best_score(self):
        return self._best_score

    @best_score.setter
    def best_score(self, value):
        if value is not None:
            if self.direction == 'maximize':
                self._best_score = max(self._best_score, value)
            else:
                self._best_score = min(self._best_score, value)

    def _get_params(self, trial):
        params = {
            name: trial._suggest(name, distribution)
            for name, distribution in self.param_distributions.items()
        }
        return params

    @staticmethod
    def get_model_iterations(cb_model):
        iterations = cb_model.get_param('iterations')
        if iterations is None:
            iterations = 1000
        return iterations

    def eval_model(self, cb_model, val_pool, metric):
        score = cb_model.eval_metrics(val_pool, metrics=metric, ntree_start=self.get_model_iterations(cb_model) - 1)
        return score[metric][0]

    def _cross_val_score(self, model, trial):
        validator = CrossValidator(model, self.x, scoring=self.scoring, y=self.y, cv=self.cv, optuna_trial=trial,
                                   n_folds_start_prune=self.n_folds_start_prune, weight_column=self.weight_column)
        if self.parallel:
            return np.mean(
                validator.parallel_fit(available_gpus=self.parallel_available_gpus)[self.scoring])
        else:
            return np.mean(
                stopit_after_timeout(self.trial_timeout, raise_exception=True)(validator.fit)()[self.scoring])

    def __call__(self, trial):
        if callable(self.param_distributions):
            params = self.param_distributions(trial)
        else:
            params = self._get_params(trial)
        if callable(self.params_post_processing):
            params = self.params_post_processing(params, trial)
        model = self.model.copy()
        try:
            result = self._cross_val_score(model.set_params(**params), trial)
        except TimeoutError:
            raise TrialPruned('Trial was pruned due to timeout')
        except Exception as e:
            if self.error_handling == 'raise':
                raise
            raise TrialPruned(f'Trial was pruned due to error: {e}')
        self.best_score = max(self.best_score, result) if self.direction == 'maximize' else min(self.best_score, result)
        return result
