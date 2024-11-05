from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def lr_model(X_train, y_train, SEED):
    """
    Fits lr model and returns best lr model
    """
    # LogisticRegression
    from sklearn.linear_model import LogisticRegression

    # Define the parameter grid
    params_lr = {
        'C': [0.03,0.04, 0.05,0.06],  # Inverse of regularization strength
        'penalty': ['l1'],  # Regularization type
        'solver': ['liblinear', 'sag'],  # Optimization algorithm
        'max_iter': [25,50],  # Maximum number of iterations for solvers
        'class_weight': ['balanced'],  # Class weights
    }

    # Example usage with GridSearchCV
    lr = LogisticRegression(random_state = SEED, class_weight={0: 1, 1: 3})

    grid_lr = GridSearchCV(estimator = lr,
                        param_grid=params_lr,
                        scoring='precision',
                        cv=10,
                        n_jobs=-1,
                        verbose=1)

    # Fit GridSearchCV to the data
    grid_lr.fit(X_train, y_train)

    best_model_lr = grid_lr.best_estimator_

    # Get the best parameters
    best_params = grid_lr.best_params_
    print("Best parameters found: ", best_params)

    return best_model_lr