import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Machine Learning App')

uploaded_file = st.sidebar.file_uploader("Select CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    st.subheader('Overview of dataset')
    overview = []
    for col in df.columns:
        row = [col, len(df[col]), str(df[col].dtype)]
        overview.append(row)
    
    st.write(pd.DataFrame(overview, columns=['Column', 'Count', 'Dtype']))
    # Select model type
    model_options = ['Linear Regression', 'Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree', 'Recommendation']
    model_type = st.sidebar.selectbox('Select model type', model_options)
    match model_type:
        case 'Linear Regression':
            model = LinearRegression()
        case 'Logistic Regression':
            model = LogisticRegression()
        case 'KNN':
            k = st.sidebar.number_input("Select K", 1, len(df))
            model = KNeighborsClassifier(n_neighbors=k)
        case 'Random Forest':
            n_estimators = st.sidebar.number_input("Select n_estimators", 1, 100)
            model = RandomForestClassifier(n_estimators)
            tree_index = st.sidebar.number_input("Select Tree Index", 0, n_estimators-1, step=1)
        case 'Decision Tree':
            model = DecisionTreeClassifier()
    target_variable = ''
    independent_variables = ''
    features = ''
    visualization_var = ''
    if model_type == 'Recommendation':
        top_n = st.sidebar.number_input("Select top n", 5, len(df))
        features = st.sidebar.multiselect('Select features', 
                                                    df.columns, 
                                                    default=list(df.columns))
    else:
        # Select independent and target variables
        target_variable = st.sidebar.selectbox('Select target variable', df.columns, index=len(df.columns)-1)
        independent_variables = st.sidebar.multiselect('Select independent variables', 
                                                    df.columns, 
                                                    default=list(df.columns.drop(target_variable)))
        
        # Select two variables for visualization
        visualization_var = st.sidebar.multiselect('Select two variables for visualization', 
                                                df.columns, 
                                                max_selections=2)
    
    # Handle missing values
    missing_values = df.isnull().sum()
    st.sidebar.write('Missing values:', missing_values)

    missing_value_options = ['Drop N/A', 'Replace with Mean', 'Replace with Median', 'Replace with Mode']
    selected_option = st.sidebar.selectbox('Select option to handle missing values', missing_value_options)

    if selected_option == 'Drop N/A':
        df.dropna(inplace=True)
    elif selected_option == 'Replace with Mean':
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col].fillna(df[col].mean(), inplace=True)
    elif selected_option == 'Replace with Median':
        for col in df.columns:
            if df[col].dtype != 'object':
                df[col].fillna(df[col].median(), inplace=True)
    elif selected_option == 'Replace with Mode':
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    if features:
        X = df[features].copy()
    elif independent_variables:
        X = df[independent_variables].copy()
    else:
        exit()
    
    # Label encode categorical features
    le = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = le.fit_transform(X[col])
    
    # Enter data used for prediction
    input = []
    for col in X.columns:
        if df[col].dtype == 'object':
            val = st.selectbox(f"{col}", df[col].unique().tolist())
        else:
            val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
        input.append(val)

if st.sidebar.button('DONE'):
    df.reset_index(inplace=True, drop=True)
    st.title(model_type)
    

    # Split data to train and test
    scaler = StandardScaler()
    if target_variable:
        X_train, X_test, y_train, y_test = train_test_split(X, df[target_variable], test_size=0.2, random_state=42)

        # Scale X data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model.fit(X_train, y_train)
    else:
        X[features] = scaler.fit_transform(X[features])

    input_df = pd.DataFrame([input], columns=X.columns)
    
    # Label encode categorical features of user input
    for col in X.columns:
        if df[col].dtype == 'object':
            le.fit_transform(df[col])
            input_df[col] = le.transform(input_df[col])
            
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Result
    if model_type != 'Recommendation':
        y_predict = model.predict(input_scaled)
        st.subheader("Result")
        st.write(f"{target_variable}: {y_predict[0]}")
    elif model_type == 'Recommendation':
        st.subheader('Similar rows in the dataset')
        similarity = cosine_similarity(input_scaled, X[features].values)
        df_copy = df.copy()
        df_copy['similarity'] = similarity[0]
        similar = df_copy.sort_values(by='similarity', ascending=False)[:top_n+1]
        st.write(similar)
        
    # Evaluation
    if model_type in ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest'] and len(df[target_variable].unique()) == 2:
        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
        precision = precision_score(y_test, model.predict(X_test), average='weighted') * 100
        recall = recall_score(y_test, model.predict(X_test), average='weighted') * 100
        f1 = f1_score(y_test, model.predict(X_test), average='weighted') * 100

        st.subheader('Performance Metrics')
        st.write(f"Accuracy: {accuracy:.2f} %")
        st.write(f"Precision: {precision:.2f} %")
        st.write(f"Recall: {recall:.2f} %")
        st.write(f"F1-score: {f1:.2f} %")
        
        # ROC and AUC
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        st.subheader('Receiver Operating Characteristic (ROC)')
        st.write(f"AUC: {roc_auc:.2f}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

        # PR and AUC-PR
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        st.subheader('Precision-Recall Curve')
        st.write(f"AUC-PR: {pr_auc:.2f}")

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AUC-PR = %0.2f)' % pr_auc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt.gcf())

        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())
      
    elif model_type in ['Linear Regression', 'Logistic Regression'] and len(df[target_variable].unique()) > 2:
        st.subheader('Evaluation')
        mse = mean_squared_error(y_test, model.predict(X_test))
        r2 = r2_score(y_test, model.predict(X_test))
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R^2 Score: {r2:.2f}")

    # Equation
    if model_type == 'Logistic Regression':
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        equation = f"{target_variable} = "
        for i, coef in enumerate(coefficients):
            equation += f"({coef:.2f} * {independent_variables[i]}) + "
        equation += f"{intercept:.2f}"
        st.subheader('Equation')
        st.write(equation)
    if model_type == 'Linear Regression':
        coefficients = model.coef_
        intercept = model.intercept_
        equation = f"{target_variable} = "
        for i, coef in enumerate(coefficients):
            equation += f"({coef:.2f} * {independent_variables[i]}) + "
        equation += f"{intercept:.2f}"
        st.subheader('Equation')
        st.write(equation)
        
    # Draw scatter plot for linear regression
    if model_type == 'Linear Regression' and len(independent_variables) == 1:
        st.subheader('Scatter Plot for Linear Regression')
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color='blue', label='Original data')
        plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression line')
        plt.xlabel(independent_variables[0])
        plt.ylabel(target_variable)
        plt.title(f'{independent_variables[0]} vs {target_variable}')
        plt.legend()
        st.pyplot(plt.gcf())
    
    # Draw tree
    if model_type == 'Decision Tree':
        st.subheader('Decision Tree Visualization')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        plot_tree(model, ax=ax, feature_names=independent_variables, filled=True, precision=2,  rounded=True, class_names=list(map(str, df[target_variable].unique().tolist())))
        st.pyplot(fig)
        tree_text = export_text(model, feature_names=independent_variables, decimals=3, show_weights=True)
        st.text(tree_text)
    if model_type == 'Random Forest':
        st.subheader('Random Forest Tree Visualization')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
        plot_tree(model.estimators_[tree_index], ax=ax, feature_names=independent_variables, filled=True, class_names=list(map(str, df[target_variable].unique().tolist())))
        st.pyplot(fig)
        tree_text = export_text(model.estimators_[tree_index], feature_names=independent_variables)
        st.text(tree_text)
        
    # Nearest neigbors
    if model_type == 'KNN':
        # Predict using the trained KNN model
        y_predict = model.predict(input_scaled)
        # Find the indices of the nearest neighbors
        _, indices = model.kneighbors(input_scaled)
        # Get the actual nearest neighbors from the training data
        nearest_neighbors = X.iloc[indices[0]]
        # Display the nearest neighbors
        st.subheader('Nearest Neighbors')
        st.write(nearest_neighbors)

        
    # Visualization 
    if len(visualization_var) == 2:
        st.subheader(f'{visualization_var[0]} and {visualization_var[1]} Visualization')
        plt.figure(figsize=(8, 6))
        plt.scatter(df[visualization_var[0]], df[visualization_var[1]], color='green')
        plt.xlabel(visualization_var[0])
        plt.ylabel(visualization_var[1])
        plt.title(f'{visualization_var[0]} vs {visualization_var[1]}')
        st.pyplot(plt.gcf())

        for col in visualization_var:
            plt.figure(figsize=(8, 6))
            plt.boxplot(df[col])
            plt.xlabel(col)
            plt.ylabel('Values')
            plt.title(f'Box Plot of {col}')
            st.pyplot(plt.gcf())