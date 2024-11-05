# KNN Model
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV

def knn_model(X_train, y_train):
    """
    Fits knn model using GridSearchCV and returns best model
    """
    params_knn = {
        'n_neighbors': [1,2,3,4],  # Number of neighbors to test
        'weights': ['uniform', 'distance'],  # Weights options
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm options
        'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
        'p': [1, 2],  # Power parameter for Minkowski distance
    }

    knn = KNN()
    grid_knn = GridSearchCV(estimator = knn,
                        param_grid=params_knn,
                        scoring='precision',
                        cv=10,
                        n_jobs=-1,
                        verbose=1)
                            
    grid_knn.fit(X_train, y_train)
    best_params_knn = grid_knn.best_params_
    best_model_knn = grid_knn.best_estimator_

    print("Best parameters found: ", best_params_knn)

    return best_model_knn