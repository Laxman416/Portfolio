from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

def rf_model(X_train, y_train, SEED):
    """
    Fits rf model and returns best rf model
    """
    params_rf = {
        'n_estimators': [75,100, 125],  # Number of trees in the forest
        'max_depth': [3,4,5, 6, 7],  # Depth of the tree
        'min_samples_leaf': [0.02, 0.04, 0.06],  # Minimum samples per leaf
    }

    rf = RandomForestClassifier(random_state=SEED)
    grid_rf = GridSearchCV(estimator = rf,
                        param_grid=params_rf,
                        scoring='precision',
                        cv=10,
                        n_jobs=-1,
                        verbose=1)

    grid_rf.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_rf.best_params_
    print("Best parameters found: ", best_params)

    best_model_rf = grid_rf.best_estimator_

    # Get feature importances
    importances = best_model_rf.feature_importances_

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance Score')
    plt.title('Feature Importances')
    plt.show()

    return best_model_rf