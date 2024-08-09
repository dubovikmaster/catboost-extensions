# Catboost-extensions

---

This library provides an easy-to-use interface for hyperparameter tuning of CatBoost models using Optuna. The `OptunaTuneCV` class simplifies the process of defining parameter spaces, configuring trials, and running cross-validation with CatBoost.

## Installation

To install the library, use pip:

```bash
pip install catboost-extensions
```

## Quick Start Guide
### OptunaTuneCV

Here is an example of how to use the library to tune a [CatBoost](https://catboost.ai/en/docs/) model using [Optuna](https://optuna.org/):

#### 1. Import necessary libraries

```python
from pprint import pprint

import pandas as pd

from catboost_extensions import (
    OptunaTuneCV, 
    CatboostParamSpace,
)
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing
import optuna
```

#### 2. Load and prepare your data

```python
# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
```

#### 3. Define your CatBoost model

```python
model = CatBoostRegressor(verbose=False, task_type='CPU')
```

#### 4. Define the parameter space

The `CatboostParamSpace` class allows you to define a parameter space for your CatBoost model. You can remove parameters that you don't want to tune using the `del_params` method.

```python
param_space = CatboostParamSpace(params_preset='general', task_type='CPU')
param_space.del_params(['depth', 'l2_leaf_reg'])
pprint(param_space.get_params_space())
```
Out:
```python
{'bootstrap_type': CategoricalDistribution(choices=('Bayesian', 'MVS', 'Bernoulli', 'No')),
 'grow_policy': CategoricalDistribution(choices=('SymmetricTree', 'Depthwise', 'Lossguide')),
 'iterations': IntDistribution(high=5000, log=False, low=100, step=1),
 'learning_rate': FloatDistribution(high=0.1, log=True, low=0.001, step=None),
 'max_bin': IntDistribution(high=512, log=False, low=8, step=1),
 'random_strength': FloatDistribution(high=10.0, log=True, low=0.01, step=None),
 'rsm': FloatDistribution(high=1.0, log=False, low=0.01, step=None),
 'score_function': CategoricalDistribution(choices=('Cosine', 'L2'))}
```
Also you can change the default values of the parameters:
```python
param_space.iterations=(1000, 2000)
```

#### 5. Set up the `OptunaTuneCV` objective

The `OptunaTuneCV` class helps to define an objective function for Optuna. You can specify the CatBoost model, the parameter space, the dataset, and other options such as the trial timeout and the scoring metric.

```python
objective = OptunaTuneCV(model, param_space, X, y, trial_timeout=10, scoring='r2')
```

#### 6. Create an Optuna study and optimize

You can choose an Optuna sampler (e.g., `TPESampler`) and then create a study to optimize the objective function.

```python
sampler = optuna.samplers.TPESampler(seed=20, multivariate=True)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=10)
```

#### 7. View the results

After the study completes, you can analyze the results to see the best hyperparameters found during the optimization.

```python
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

## Contributing

If you want to contribute to this library, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.
