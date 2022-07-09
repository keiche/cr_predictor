# Overview

Predicting challenge ratings of fifth edition monsters via supervised machine learning. This can be used to help create homebrew monsters with the appropriate challenge rating.

# Machine Learning

## Feature Set

All features are represented by integers for use in the ML model.

| Feature        | Description                                             |
|----------------|---------------------------------------------------------|
| `name`         | Creature name                                           |
| `ac`           | Armor Class                                             |
| `hp`           | Health Points (max)                                     |
| `hit_bonus`    | Bonus "to hit" value                                    |
| `max_dmg_turn` | Max damage that can be done in a single turn (no buffs) |
| `legendary`    | Legendary creature (Boolean)                            |

## Regression Model

The "Extremely Randomized Trees" regression model was chosen as it has the highest accuracy.

| Model                      | sklearn Function                          | Cross-Validation<br>(10 splits) |
|----------------------------|-------------------------------------------|---------------------------------|
| Extremely Randomized Trees | `ensemble.ExtraTreesRegressor()`          | 95.00%                          |
| Partial Least Squares      | `cross_decomposition.PLSRegression()`     | 94.56%                          |
| Linear                     | `linear_model.LinearRegressor()`          | 94.46%                          |
| Bayesian                   | `linear_model.BayesianRidge()`            | 94.35%                          |
| Gradient Tree Boosting     | `ensemble.GradientBoostingRegressor()`    | 94.17%                          |
| Random Forest              | `ensemble.RandomForestRegressor()`        | 94.15%                          |
| Bagging Meta-Estimator     | `ensemble.BaggingRegressor()`             | 93.44%                          |
| Decision Tree              | `tree.DecisionTreeRegressor(max_depth=5)` | 90.53%                          |
| Extra Decision Trees       | `tree.ExtraTreeRegressor()`               | 90.33%                          |
| Quantile                   | `linear_model.QuantileRegressor()`        | 90.10%                          |
| Multi-layer Perceptron     | `neural_network.MLPRegressor()`           | 85.72%                          |
| Support Vector Machine     | `svm.SVR()`                               | 84.97%                          |

# Install

Several options are available

## Poetry

```shell
poetry update
poetry install
```

## Docker

```shell
docker build . -t cr_predict:latest
```

## Wheel

```shell
poetry build
pip3 install dist/cr_predictor*.whl
```

# Usage

```
Usage: cr_predictor [OPTIONS] COMMAND [ARGS]...

Options:
  -d, --debug  Debug output
  --help       Show this message and exit.

Commands:
  feature_importances  Cross-validation scores for a feature set
  predict              Predict the CR using a classification model
  predict_known        Predict the CR using a classification model with...
  save_model           Fit and save a classification model
  validate             Cross-validation scores for a feature set
```

# Commands

## train
Train and save the model to a file

Usage
```
Usage: cr_predictor save_model [OPTIONS] FEATURESET

  Fit and save a classification model

Options:
  -o, --output TEXT  Output model name  [default: cr_model.joblib]
  --help             Show this message and exit.
```

Example
```
$ cr_predictor save_model srd_training.csv
Saved model to cr_model.joblib
```
or with Docker
```shell
docker run --rm -v $(pwd)/training_set:/app/training_set cr-predictor:latest train -o training_set/srd_model.joblib training_set/srd_training.csv
```

## predict
Predict CRs using the trained model

Usage
```
Usage: cr_predictor predict [OPTIONS] MODEL PREDICTION

  Predict the CR using a classification model

Options:
  --help  Show this message and exit.
```
Command
```shell
echo "monster1,14,52,6,14,0" > homebrew.csv
echo "monster2,18,180,10,34,0" >> homebrew.csv
cr_predictor predict cr_model.joblib homebrew.csv
```
Output
```
+----------+--------------+
|   name   | predicted cr |
+----------+--------------+
| monster1 |      3       |
| monster2 |      14      |
+----------+--------------+
```

## validate
Validate the training set

Usage
```
Usage: cr_predictor validate [OPTIONS] FEATURESET

  Cross-validation scores for a feature set

Options:
  -s, --split INTEGER   Number of cross-validation splits  [default: 10]
  -t, --testsize FLOAT  Percent of training set to use for testing (decimal
                        from 0-1)  [default: 0.3]
  --help                Show this message and exit.

```
Command
```commandline
cr_predictor validate srd_training.csv
```
Output
```
+------------+-------+
| test count | score |
+------------+-------+
|     0      | 96.54 |
|     1      | 94.55 |
|     2      | 93.15 |
|     3      | 95.94 |
|     4      | 96.23 |
|     5      | 95.53 |
|     6      | 95.93 |
|     7      | 93.88 |
|     8      | 96.55 |
|     9      | 91.67 |
+------------+-------+
95.0% accuracy with a standard deviation of 0.0156
```

## predict_known
Predicting with known CR. Using the `validate` command is much more accurate, but this provides some fun granular data.

Usage
```
Usage: cr_predictor predict_known [OPTIONS] MODEL PREDICTION

  Predict the CR using a classification model with known outcomes Used to show
  granular test output for a trained model

Options:
  --help  Show this message and exit.
```
Command
```commandline
cr_predictor predict_known cr_model.joblib predict_known.csv
```
Output (trained without these monsters for a demonstration)
```
+-----------------------+--------+-----------+------------+                                                    
|         name          | actual | predicted | difference |
+-----------------------+--------+-----------+------------+
|  ancient gold dragon  |   24   |    23     |    1.0     |
|   adult gold dragon   |   17   |    16     |    1.0     |
|   young gold dragon   |   10   |    10     |    0.0     |
|  gold dragon wymling  |   3    |     3     |    0.0     |
| ancient silver dragon |   23   |    23     |    0.0     |
|  adult silver dragon  |   16   |    16     |    0.0     |
|  young silver dragon  |   9    |     9     |    0.0     |
| silver dragon wymling |   2    |     2     |    0.0     |
|        vampire        |   13   |    10     |    3.0     |
+-----------------------+--------+-----------+------------+

Difference Stats

+--------+-------+
|  stat  | value |
+--------+-------+
|  avg   | 0.56  |
| median |  0.0  |
|  max   |  3.0  |
|  min   |  0.0  |
|  std   | 0.96  |
+--------+-------+
```

## feature
Validating feature importance

Usage
```
Usage: cr_predictor feature_importances [OPTIONS] FEATURESET

  Cross-validation scores for a feature set

Options:
  --help  Show this message and exit.
```
Command
```commandline
cr_predictor features srd_training.csv
```
Output
```
+-----+--------------+------------+                                                            
| num |     name     | importance |
+-----+--------------+------------+
|  0  |      ac      |   0.0309   |
|  1  |      hp      |   0.2409   |
|  2  |  hit_bonus   |   0.2665   |
|  3  | max_dmg_turn |   0.1803   |
|  4  |  legendary   |   0.2815   |
+-----+--------------+------------+
```

# Thoughts

* Current feature set
  * The current feature set works well for monsters that do purely damage. This feature set does not capture non-damage abilities well (status conditions, rejuvenation, pack tactics, etc.) 
    * Monsters such as the "Succubus / Incubus" or "Vampire" are outliers in the feature set as they rely on effects other than pure damage.
    * It would be interesting to think of other features to capture to improve this model.
  * Data collection and cleaning is the longest part of any data science project, so for now I won't be exploring other features. Feel free to test it out to see if it changes anything!
* The "Systems Reference Document" data set
  * This is a large data set that is available via the "Open Gaming License". A larger data set that has a greater mix of CRs may improve the model.
  * There are CRs with a very low amount of examples (4, 7, 12, 19)
* Caveats
  * To calculate the `max_dmg_turn` value, I took the raw damage number that was presented, not the highest possible dice value. Example: if a creature attacks "bites for 5 (1d6 + 2) piercing damage", then I said `max_dmg_turn` was 5.  
  * The Lich's ability to cast `Power Word Kill` is hard to quantify for `max_dmg_turn`, so I used `Finger of Death` instead.

## Future

* Additional [metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) to understand and improve the fit

# License

The [prediction training set](training_set/srd_training.csv) uses data from the [Systems Reference Document 5.1](https://media.wizards.com/2016/downloads/DND/SRD-OGL_V5.1.pdf) which contains content under the [Open Gaming License](OGL.md)
