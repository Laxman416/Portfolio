from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def gbc_model(X_train, y_train, SEED):
    """
    fits gbc model and returns best model
    """
    params_gbc = {
        'n_estimators': [200], 
        'learning_rate': [0.01, 0.02],  
        'max_depth': [3, 5], 
        'subsample': [0.8, 1.0],  
    }

    # Initialize Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(random_state=SEED)

    grid_gbc = GridSearchCV(estimator=gbc,
                            param_grid=params_gbc,
                            scoring='precision',
                            cv=10,
                            n_jobs=-1,
                            verbose=1)

    # Fit GridSearchCV to the data
    grid_gbc.fit(X_train, y_train)

    # Retrieve the best estimator and parameters
    best_model_gbc = grid_gbc.best_estimator_
    best_params_gbc = grid_gbc.best_params_

    print("Best parameters for GBC: ", best_params_gbc)

    return best_model_gbc