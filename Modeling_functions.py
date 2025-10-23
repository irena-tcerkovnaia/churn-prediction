def check_overfitting_classification(model, X_train, y_train, X_test, y_test, metric_fun):
    """
   check overfitting for classification

    Parameters:
    - model: Trained classification model.
    - X_train: Training features.
    - y_train: Training target labels.
    - X_test: Test features.
    - y_test: Test target labels.
    - metric_fun: Metric function (e.g., accuracy_score, f1_score, roc_auc_score).
    """
    # Predict on training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metric values
    value_train = metric_fun(y_train, y_pred_train)
    value_test = metric_fun(y_test, y_pred_test)

    # Print results
    print(f'{metric_fun.__name__} train: %.3f' % value_train)
    print(f'{metric_fun.__name__} test: %.3f' % value_test)
    print(f'delta = {(abs(value_train - value_test) / value_test * 100):.1f} %')

# Catboost for Classification

def objective_lgb(trial, X, y, N_FOLDS, random_state, cat_feat):
        params = {

            "n_estimators": trial.suggest_categorical("n_estimators", [300]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.08797829241393999]),
            #         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "l2_leaf_reg": trial.suggest_uniform("l2_leaf_reg", 1e-5, 1e2),
            'random_strength': trial.suggest_uniform('random_strength', 10, 50),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS", "No"]),
            'border_count': trial.suggest_categorical('border_count', [128, 254]),
            'grow_policy': trial.suggest_categorical('grow_policy', ["SymmetricTree", "Depthwise", "Lossguide"]),

            'od_wait': trial.suggest_int('od_wait', 500, 2000),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
            # "cat_features": trial.suggest_categorical("cat_features", ["cat_features"])
            # "loss_function": trial.suggest_categorical("loss_function", ["MAE"]),
            "use_best_model": trial.suggest_categorical("use_best_model", [True]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["AUC"]),
            "random_state": random_state
        }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 100)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float(
                "subsample", 0.1, 1, log=True)

        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RAND)

        cv_predicts = np.empty(N_FOLDS)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_data = Pool(data=X_train, label=y_train, cat_features=cat_feat)
            eval_data = Pool(data=X_test, label=y_test, cat_features=cat_feat)

            model = CatBoostClassifier(**params)
            model.fit(train_data,
                      eval_set=[eval_data],
                      early_stopping_rounds=100,
                      verbose=0)

            preds = model.predict(X_test)
            cv_predicts[idx] = roc_auc_score(y_test, preds)

        return np.mean(cv_predicts)

    # %%
def cross_validation_cat(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             clf,
                             params: dict,
                             cat_features: list = None,
                             eval_metric: str = None,
                             early_stop: bool = False,
                             early_stopping_rounds: int = 100,
                             num_folds: int = 3,
                             random_state: int = 10,
                             shuffle: bool = True):

        """using cross-validation for classification problem (catboost base algorithms - no tuning)"""

        # shuffle - shuffle data before splitting only
        folds = KFold(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
        score_oof = []
        predictions_test = []

        for fold, (train_index,
                   test_index) in enumerate(folds.split(X_train, y_train)):
            X_train_, X_val = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_, y_val = y_train.iloc[train_index], y_train.iloc[test_index]
            y_val_exp = np.exp(y_val) - 1

            model = clf(**params)

            if early_stop == True:
                if eval_metric is None:
                    model.fit(X_train_,
                              y_train_,
                              eval_set=[(X_val, y_val)],
                              cat_features=cat_features,
                              silent=True,
                              early_stopping_rounds=early_stopping_rounds)
                else:
                    model.fit(X_train_,
                              y_train_,
                              eval_set=[(X_val, y_val)],
                              eval_metric=eval_metric,
                              silent=True,
                              cat_features=cat_features,
                              early_stopping_rounds=early_stopping_rounds)
            else:
                model.fit(X_train_, y_train_, cat_features=cat_features)

            y_pred_val = model.predict(X_val)
            y_pred = model.predict(X_test)

            print(
                "Fold:", fold + 1,
                         "ROC-AUC SCORE %.3f" % roc_auc_score(y_val, y_pred_val))
            print("---")

            # oof list
            score_oof.append(roc_auc_score(y_val, y_pred_val))
            # holdout list
            predictions_test.append(y_pred)

        return score_oof, predictions_test