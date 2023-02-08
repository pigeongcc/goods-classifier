import pandas as pd
import numpy as np
import catboost as cb
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from catboost.utils import get_confusion_matrix
from typing import Tuple


params_default = {
    'iterations': 316,
    'learning_rate': 0.1442774314920995,
    'depth': 4,
    'l2_leaf_reg': 3.925290717509199,
    'boosting_type': 'Plain',
    'eval_metric': 'TotalF1:average=Macro',
    'task_type': 'GPU'
}

metrics_default = ['Accuracy', 'TotalF1:average=Weighted', 'TotalF1:average=Macro']

finetune_params_example = {
    'iterations': (300, 1000),
    'learning_rate': (0.005, 0.15),
    'depth': (3, 4),
    'l2_leaf_reg': (1, 10)
}

class Model:
    def __init__(self,
                 cat_features: list,
                 text_features: list,
                 target: str,
                 params: dict = None,
                 finetune: bool = False,
                 finetune_params: dict = None,
                 finetune_n_trials: int = None,
                 metrics: list = None
                 ):
        """
        :param cat_features: categorical features to use while training
        :param text_features: textual features to use while training
        :param target: name of the target column
        :param params: hyperparameters of the model. Default ones are defined in params_default
        :param finetune: flag whether to use finetuning
        :param finetune_params: dict with ranges of hyperparameters. See example in finetune_params_example
        :param finetune_n_trials: number of trials to do while finetuning
        :param metrics: metrics to print after the learning.
        """
        self.cat_features = cat_features
        self.text_features = text_features
        self.features = self.cat_features + self.text_features
        self.target = target
        if params is None:
            self.params = params_default
        else:
            self.params = params
        self.finetune = finetune
        if finetune_params is not None:
            self.finetune_params = finetune_params
        self.finetune_n_trials = finetune_n_trials
        if metrics is None:
            self.metrics = metrics_default
        else:
            self.metrics = metrics
        self.metrics_values = None
        self.learn_pool = self.test_pool = self.model = None

    def train(self, data: pd.DataFrame) -> cb.CatBoostClassifier:
        self.learn_pool, self.test_pool = self.preprocess_data(data)

        if self.finetune:
            finetuned_params = self.finetuning()
            for key, value in finetuned_params.items():
                self.params[key] = value

        self.model = cb.CatBoostClassifier(**self.params, random_seed=42)
        self.model.fit(self.learn_pool, eval_set=self.test_pool, verbose=100)

        self.metrics_values = self.model.eval_metrics(data=self.test_pool, metrics=self.metrics)

        return self.model

    def finetuning(self) -> dict:
        def get_trial_params(trial):
            return {
                'iterations': trial.suggest_int('iterations', self.finetune_params['iterations'][0],
                                                self.finetune_params['iterations'][1]),
                'learning_rate': trial.suggest_float('learning_rate', self.finetune_params['learning_rate'][0],
                                                     self.finetune_params['learning_rate'][1]),
                'depth': trial.suggest_int('depth', self.finetune_params['depth'][0],
                                             self.finetune_params['depth'][1]),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', self.finetune_params['l2_leaf_reg'][0],
                                                   self.finetune_params['l2_leaf_reg'][1]),
                'boosting_type': trial.suggest_categorical('boosting_type', [self.params['boosting_type']]),
                'eval_metric': self.params['eval_metric'],
                'task_type': self.params['task_type']
            }

        def objective(trial):
            params = get_trial_params(trial)
            model = cb.CatBoostClassifier(**params, random_seed=42)
            model.fit(self.learn_pool, eval_set=self.test_pool, verbose=0)

            metric = self.params['eval_metric']
            score = model.eval_metrics(data=self.test_pool, metrics=metric)[metric]
            score = score[len(score)-1]

            return score

        sampler = TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=self.finetune_n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return trial.params

    def predict(self, data: cb.Pool):
        pass

    def get_pool(self, X: pd.DataFrame, y: pd.Series) -> cb.Pool:
        return cb.Pool(
            X,
            y,
            cat_features=self.cat_features,
            text_features=self.text_features,
            feature_names=self.features
        )

    def get_pool_from_df(self, data: pd.DataFrame) -> cb.Pool:
        X = data[self.features]
        y = data[self.target]
        return self.get_pool(X, y)

    def preprocess_text_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for cat_feature in self.cat_features:
            data[cat_feature] = data[cat_feature].fillna('')
        for text_feature in self.text_features:
            data[text_feature] = data[text_feature].fillna('')
        data = data.replace(to_replace=float('nan'), value="")
        return data

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[cb.Pool, cb.Pool]:
        data = self.preprocess_text_features(data)

        X = data[self.cat_features + self.text_features]
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
        learn_pool = self.get_pool(X_train, y_train)
        test_pool = self.get_pool(X_test, y_test)
        return learn_pool, test_pool

    def print_metrics(self):
        print("Resulting Metrics:")
        for key, val in self.metrics_values.items():
            print(f"{key}: {val[-1]}")

    def print_feature_importances(self):
        print("Feature Importances:")
        for name, importance in zip(self.model.feature_names_, self.model.feature_importances_):
            print(name, '\t', importance)

    def get_confusion_matrix(self, data: pd.DataFrame = None) -> np.ndarray:
        # if data is None, self.test_pool is used to costruct the CM
        # else, some data is passed and is going to be transformed into a cb.Pool
        if data is None:
            pool = self.test_pool
        else:
            pool = self.get_pool_from_df(data)
        return get_confusion_matrix(self.model, pool)

    def get_class_name(self, idx) -> str:
        return -1 if idx == -1 else self.model.classes_[idx]

    def get_inaccurate_classes(self, data: pd.DataFrame = None) -> pd.DataFrame:
        cm = self.get_confusion_matrix(data)
        inaccurate_indices = []
        for i in range(cm.shape[0]):
            if np.sum(cm[i]) != cm[i, i] or cm[i, i] == 0:
                inaccurate_indices.append(i)

        data = {"class_id": [], "class_name": [], "accuracy": [], "num_of_examples": [], "confused_with": []}
        for idx in inaccurate_indices:
            sum_idx = np.sum(cm[idx])
            acc = cm[idx, idx] / sum_idx
            confused_with = ""
            for idx_nonzero in np.nonzero(cm[idx])[0]:
                if idx != idx_nonzero:
                    confused_with += f'"{self.get_class_name(idx_nonzero)}" (class {idx_nonzero}) {int(cm[idx, idx_nonzero])}/{int(sum_idx)}. '
            data["class_id"].append(idx)
            data["class_name"].append(self.get_class_name(idx))
            data["accuracy"].append(acc)
            data["num_of_examples"].append(np.sum(cm[idx]))
            data["confused_with"].append(confused_with)

        return pd.DataFrame(data).sort_values(by="accuracy")

    def save_model(self, filename: str, format: str = 'cbm'):
        self.model.save_model(filename, format=format)

    def load_model(self, filename: str):
        self.model = cb.CatBoostClassifier()
        self.model.load_model(filename)

    def label(self, data: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, float]:
        data = self.preprocess_text_features(data)
        data = data[self.features]
        pool = cb.Pool(
            data,
            cat_features=self.cat_features,
            text_features=self.text_features,
            feature_names=self.features
        )

        predictions = []
        probabilities = []
        predicted = nonpredicted = 0
        probas_matrix = self.model.predict_proba(pool)
        for probas in probas_matrix:
            max_ind = np.argsort(probas)[-1]
            if probas[max_ind] > threshold:
                predictions.append(max_ind)
                probabilities.append(probas[max_ind])
                predicted += 1
            else:
                predictions.append(-1)
                probabilities.append(probas[max_ind])
                nonpredicted += 1
        ratio = predicted / (predicted + nonpredicted)

        data = data.assign(pred_id=predictions)
        data["pred_name"] = ""
        data["pred_name"] = data["pred_id"].apply(self.get_class_name)
        data = data.assign(pred_proba=probabilities)

        return data, ratio


