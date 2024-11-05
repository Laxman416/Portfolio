# DecisionClassiifciation Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def dt_model(X_train, y_train, SEED):
    """
    Fits dt model and returns best dt model
    """
    params_dt = {
        'max_depth': [None, 5],
        'min_samples_split': [4, 5, 6],  
        'min_samples_leaf': [2, 3],  
        'max_features': ['log2', 'sqrt'],  
        'class_weight': ['balanced'],  # Class weights
    }

    dt = DecisionTreeClassifier(random_state=SEED)

    grid_dt = GridSearchCV(estimator=dt,
                        param_grid=params_dt,
                        scoring='precision',
                        cv=10,
                        n_jobs=-1,
                        verbose=1)

    grid_dt.fit(X_train, y_train)

    best_model_dt = grid_dt.best_estimator_
    best_params_dt = grid_dt.best_params_

    print("Best parameters found for Decision Tree: ", best_params_dt)
    
    return best_model_dt