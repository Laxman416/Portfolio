from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def svc_model(X_train, y_train, SEED):
    """
    Fits svc model and returns best model
    """
    params_svc = {
        'C': [0.7],  # Regularization parameter
        'kernel': ['linear'],  # Kernel type
        'gamma': ['scale']  
    }

    svc = SVC(probability=True, random_state=SEED)

    grid_svc = GridSearchCV(estimator=svc,
                            param_grid=params_svc,
                            scoring='precision',
                            cv=10,
                            n_jobs=-1,
                            verbose=2)

    # Fit GridSearchCV to the data
    grid_svc.fit(X_train, y_train)

    best_model_svc = grid_svc.best_estimator_
    best_params_svc = grid_svc.best_params_

    print("Best parameters for SVC: ", best_params_svc)

    return best_model_svc