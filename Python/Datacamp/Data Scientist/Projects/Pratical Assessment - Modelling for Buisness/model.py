from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from knn_model import knn_model
from lr_model import lr_model
from rf_model import rf_model
from dt_model import dt_model
from svc_model import svc_model
from gbc_model import gbc_model


scaler = StandardScaler()
mm_scaler = MinMaxScaler()
clean_recipe_df = pd.read_csv('clean_recipe_df.csv')
SEED = 9
# list_to_drop = ['category_Dessert', 'category_One Dish Meal', 'category_Lunch/Snacks', 'category_Meat', 'category_Pork', 'high_traffic', 'recipe', 'protein','servings','sugar','calories','carbohydrate']
# list_to_drop = ['category_Dessert', 'category_One Dish Meal', 'category_Lunch/Snacks', 'category_Meat', 'category_Pork', 'high_traffic', 'recipe', 'protein','servings','sugar','calories','carbohydrate', 'carbohydrate_per_serving','calories_per_serving']

list_to_drop = ['servings', 'category_Dessert', 'category_One Dish Meal', 'category_Lunch/Snacks', 'category_Meat', 'calories', 'sugar', 'carbohydrate', 'category_Pork', 'recipe','high_traffic'] # Gives accuracy of 0.80

# list_to_drop = ['recipe', 'calories', 'carbohydrate', 'sugar', 'protein', 'category_Beverages', 'sugar_per_serving', 'carbohydrate_per_serving', 'calories_per_serving', 'protein_per_serving', 'high_traffic'] # From RFE

nutritional_features = ['sugar','carbohydrate','calories','protein']

def pre_formating(df, list_nutritional_features):
    """
    Function to create dummy variables and create columns out of the existing features.
    New columns will be created for the nutrional features divide per serving
    """
    # Creates dummy variables
    df = pd.get_dummies(df, columns=['category'], prefix='category')


    for feature in nutritional_features.copy():
        column_name = f'{feature}_per_serving'
        df[column_name] = df[feature] / df['servings']
        list_nutritional_features .append(column_name)

    return df

def format_data_to_model(df, dropped_list):

    X_df = df.drop(dropped_list, axis=1)
    y_df = df['high_traffic']

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                       y_df,
                                       test_size = 0.2, # size of test data
                                       random_state = SEED,
                                       stratify=y_df)

    return X_train, X_test, y_train, y_test

def transform_data(df, scaler, min_max_scaler, test=False):
    """
    Transform the input DataFrame by applying log transformations, 
    dummy categorical variables, and scaling to specific features.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to transform.
    test (bool): A flag indicating whether the DataFrame is for testing 
                 (True) or training (False). If True, uses 
                 `transform()` instead of `fit_transform()` to avoid 
                 data leakage.

    Returns:
    pd.DataFrame: The transformed DataFrame with log-transformed 
                  nutritional features, dummy categorical 
                  features, and scaled values for servings.
    """ 
    #Avoids DataLeakage using test

    # Log transform
    for feature in nutritional_features:
        if feature in df.columns:  
            df[feature] = np.log1p(df[feature])
    
    available_nutritional_features = [feature for feature in nutritional_features if feature in df.columns]
    
    # Scales Data
    if test == False:
        if available_nutritional_features:  
            df[available_nutritional_features] = scaler.fit_transform(df[available_nutritional_features])
        if 'servings' in df.columns:
            df['servings'] = mm_scaler.fit_transform(df[['servings']])
    elif test == True:
        if available_nutritional_features:  
            df[available_nutritional_features] = scaler.transform(df[available_nutritional_features])
        if 'servings' in df.columns:
            df['servings'] = mm_scaler.transform(df[['servings']])
    return df


def rfe_selection(X_train_rfe, y_train_rfe, n_features_to_select):
    """
    RFE selection to select key features and to return the datasets that are used for the models.
    """
    rfe_selector = RFE(estimator=RandomForestClassifier(random_state=42), 
                       n_features_to_select=n_features_to_select, 
                       step=1)
    rfe_selector.fit(X_train_rfe, y_train_rfe)
    
    selected_features = X_train_rfe.columns[rfe_selector.support_].tolist()

    return selected_features

def RFE_code(clean_recipe_df, nutritional_features):

    clean_recipe_df = pre_formating(clean_recipe_df, nutritional_features)
    X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = format_data_to_model(clean_recipe_df, ['high_traffic'])

    X_train_rfe = transform_data(X_train_rfe, scaler, mm_scaler, False)
    X_test_rfe = transform_data(X_test_rfe, scaler, mm_scaler, True)
    list_to_drop_rfe = rfe_selection(X_train_rfe, y_train_rfe, 12)
    list_to_drop_rfe.append('high_traffic')
    print(f"RFE Selection List: {list_to_drop_rfe}")
    X_train, X_test, y_train, y_test = format_data_to_model(clean_recipe_df, list_to_drop_rfe)

    X_train = transform_data(X_train, scaler, mm_scaler, False)
    X_test = transform_data(X_test, scaler, mm_scaler, True)

    return X_train, X_test, y_train, y_test

### Working Code

def working_code(clean_recipe_df, nutritional_features, list_to_drop):
    clean_recipe_df = pre_formating(clean_recipe_df, nutritional_features)
    X_train, X_test, y_train, y_test = format_data_to_model(clean_recipe_df, list_to_drop)

    X_train = transform_data(X_train, scaler, mm_scaler, False)
    X_test = transform_data(X_test, scaler, mm_scaler, True)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = working_code(clean_recipe_df, nutritional_features, list_to_drop)
# X_train, X_test, y_train, y_test = RFE_code(clean_recipe_df, nutritional_features)


def main():
    # Ensemble Learning
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import confusion_matrix

    SEED = 9

    best_model_lr = lr_model(X_train, y_train, SEED)
    best_model_knn = knn_model(X_train, y_train)
    best_model_rf = rf_model(X_train, y_train, SEED)
    # best_model_dt = dt_model(X_train, y_train, SEED)
    best_model_svc = svc_model(X_train, y_train, SEED)
    best_model_gbc = gbc_model(X_train, y_train, SEED)

    # classifiers = [('Logistic Regression', best_model_lr)]

    classifiers = [('Logistic Regression', best_model_lr),
                ('K Nearest Neighbors', best_model_knn),
                ('Random Forest', best_model_rf),
                ('SVC', best_model_svc),
                ('Gradient Boosting Classifier', best_model_gbc)
                ]

    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{clf_name} Test Accuracy: {accuracy:.2f}")

            
    vc = VotingClassifier(estimators = classifiers, voting = 'soft')

    print("-------------------")
    vc.fit(X_train, y_train)
    y_pred_vc = vc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_vc)
    print(f"VC Accuracy: {accuracy:.2f}")

    cm_vc = confusion_matrix(y_test, y_pred_vc)
    print("Voting Classifier (VC) Confusion Matrix:\n", cm_vc)

    return

main()