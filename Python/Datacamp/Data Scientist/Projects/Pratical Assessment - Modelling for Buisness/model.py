from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from lr_model import lr_model
from rf_model import rf_model
from gbc_model import gbc_model
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

scaler = StandardScaler()
mm_scaler = MinMaxScaler()
clean_recipe_df = pd.read_csv('clean_recipe_df.csv')
SEED = 9

list_to_drop = ['servings', 'category_Dessert', 'category_One Dish Meal', 'category_Lunch/Snacks', 'category_Meat', 'calories', 'sugar', 'carbohydrate', 'category_Pork', 'recipe','high_traffic'] # Gives accuracy of 0.80, no new transformation

nutritional_features = ['sugar','carbohydrate','calories','protein']

def pre_formating(df, list_nutritional_features):
    """
    Function to create dummy variables and create columns out of the existing features.
    New columns will be created for the nutrional features divide per serving
    """
    # Creates dummy variables
    df = pd.get_dummies(df, columns=['category'], prefix='category')

    # # No additional features gets:
    
    # # RF score 0.81
    for feature in nutritional_features.copy():
        if feature != 'protein':
            column_name = f'{feature}_per_serving'
            df[column_name] = df[feature] / (df['servings'])
            list_nutritional_features.append(column_name)
            df.drop(feature, axis=1, inplace=True)

    # LR 0.80 score and VC 0.80
    # for feature in nutritional_features.copy():
    #         column_name = f'{feature}_per_serving^2'
    #         df[column_name] = df[feature] / (df['servings']**2)
    #         list_nutritional_features.append(column_name)
    #         df.drop(feature, axis=1, inplace=True)

    # # LR 0.80 score and VC 0.79
    # for feature in nutritional_features.copy():
    #     if feature != 'protein':
    #         column_name = f'{feature}_per_serving^2'
    #         df[column_name] = df[feature] / (df['servings']**2)
    #         list_nutritional_features.append(column_name)
    #         df.drop(feature, axis=1, inplace=True)

    df.drop(['servings'], axis = 1, inplace = True)

    return df

def format_data_to_model(df, dropped_list):
    """
    train_test_split and drop lists
    """
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

def RFE_code(clean_recipe_df, nutritional_features, num_features):

    clean_recipe_df = pre_formating(clean_recipe_df, nutritional_features)
    all_columns = clean_recipe_df.columns.tolist()
    X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = format_data_to_model(clean_recipe_df, ['high_traffic'])

    X_train_rfe = transform_data(X_train_rfe, scaler, mm_scaler, False)
    X_test_rfe = transform_data(X_test_rfe, scaler, mm_scaler, True)
    list_to_keep = rfe_selection(X_train_rfe, y_train_rfe, num_features)
    list_to_drop_rfe = [col for col in all_columns if col not in list_to_keep]
    # list_to_drop_rfe.append('high_traffic')
    print(f"RFE Columns to Drop: {list_to_drop_rfe}")
    X_train, X_test, y_train, y_test = format_data_to_model(clean_recipe_df, list_to_drop_rfe)

    X_train = transform_data(X_train, scaler, mm_scaler, False)
    X_test = transform_data(X_test, scaler, mm_scaler, True)

    return X_train, X_test, y_train, y_test

def save_metrics_to_txt(accuracy_dict, precision_dict, cm_dict, file_name):
    with open(file_name, 'w') as file:
        # Loop through each classifier name
        for clf_name in accuracy_dict:
            file.write(f"{clf_name}:\n")
            
            # Write Accuracy
            accuracy = accuracy_dict[clf_name]
            file.write(f"    Accuracy: {accuracy:.2f}\n")
            
            # Write Precision
            precision = precision_dict[clf_name]
            file.write(f"    Precision: {precision:.2f}\n")
            
            # Write Confusion Matrix
            cm = cm_dict[clf_name]
            file.write("\n")

            file.write(f"    Confusion Matrix:\n")
            file.write(f"    {cm[0]}    [TN, FP]\n")
            file.write(f"    {cm[1]}    [FN, TP]\n")
            file.write("\n")
    return

def main():
    # Ensemble Learning

    SEED = 9

    X_train, X_test, y_train, y_test = RFE_code(clean_recipe_df, nutritional_features, num_features=11)
    print(f"X train columns: {X_train.columns}")

    best_model_lr = lr_model(X_train, y_train, SEED)
    best_model_rf = rf_model(X_train, y_train, SEED)
    best_model_gbc = gbc_model(X_train, y_train, SEED)


    classifiers = [('Logistic Regression', best_model_lr),
                ('Random Forest', best_model_rf),
                ('Gradient Boosting Classifier', best_model_gbc)
                ]

    cm_dict = {}
    precision_score_dict = {}
    accuracy_score_dict = {}

    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_score_dict[clf_name] = accuracy
        print(f"{clf_name} Test Accuracy: {accuracy:.3f}")
        cm = confusion_matrix(y_test, y_pred)
        cm_dict[clf_name] = cm
        precision = precision_score(y_test, y_pred)
        precision_score_dict[clf_name] = precision
        print(f"{clf_name} Test Precision: {precision:.3f}")

    vc = VotingClassifier(estimators = classifiers, voting = 'soft', weights = [0.782,0.81,0.782])
    print("-------------------")
    vc.fit(X_train, y_train)
    y_pred_vc = vc.predict(X_test)
    accuracy_vc = accuracy_score(y_test, y_pred_vc)
    precision_vc = precision_score(y_test, y_pred_vc)
    cm_vc = confusion_matrix(y_test, y_pred_vc)

    # Append VC results to dictionaries
    accuracy_score_dict['Voting Classifier'] = accuracy_vc
    precision_score_dict['Voting Classifier'] = precision_vc
    cm_dict['Voting Classifier'] = cm_vc

    print(f"VC Accuracy: {accuracy_vc:.3f}")
    print(f"VC Precision: {precision_vc:.3f}")

            
    print("Voting Classifier (VC) Confusion Matrix:\n", cm_vc)

    print("Random Forest (RF) Confusion Matrix:\n", cm_dict['Random Forest'])
    print("Logistic Regression (LR) Confusion Matrix:\n", cm_dict['Logistic Regression'])

    save_metrics_to_txt(accuracy_score_dict, precision_score_dict, cm_dict, 'model_results.txt')

    return

main()