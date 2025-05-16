import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import shap

# le = LabelEncoder()

model = joblib.load('C:\\Users\\User\\Machine Learning\\Loan\\Loan_Status_model.pkl') 

st.title('LOAN STATUS')

tab,tab1 = st.tabs(['Data Analysis','Model'])

# st.header('')
with tab:
    uploaded_file = st.file_uploader('Choose a csv file',type='csv')
    

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success('file was successfully uploaded')
            st.subheader('Preview of Data')
            st.write(df.head())

            fig,ax1 = plt.subplots()
            sns.countplot(data=df,x='Gender',hue='Loan_Status')
            plt.title('Gender Count On Loan Status')
            st.pyplot(fig)

            fig1 = px.violin(df,x='Dependents',y='LoanAmount',color='Married',title='Chart of Dependency On Loan Amount With Marriage')
            st.plotly_chart(fig1)
        except Exception as e:
            st.error(f"Error: {e}")
with tab1:
    st.header('Input Feature')
    gender = st.selectbox('Gender',options=['male','female'])
    married = st.selectbox('Married',options=['Married','Unmarried'])
    dependent = st.selectbox('Dependents',options=[0,1,2,'3+'])
    education = st.selectbox('Education',options=['Graduated','Not Graduated'])
    self_employed = st.selectbox('Self_Employed',options=['No','Yes'])
    loan_amount = st.number_input('Loan Amount',min_value=0,max_value=500,value=100)
    loan_amount_term = st.slider('Loan Term',min_value=0,max_value=500,value=360)
    credit_history = st.slider('Credit History',min_value=0.0,max_value=1.0,value=0.84219858)
    property_area = st.selectbox('Property_Area',options=['Urban','Rural','Semiurban'])


    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependent,
        'Education': education,
        'Self_Employed': self_employed,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    input_df = pd.DataFrame([input_data])
    def predict_loan(input_data):
        input_df = pd.DataFrame([input_data])
        gender_map = {'male': 1, 'female': 0}
        married_map = {'Married': 1, 'Unmarried': 0}
        dependents_map = {0: 0, 1: 1, 2: 2, '3+': 3}
        education_map = {'Graduated': 1, 'Not Graduated': 0}
        self_employed_map = {'Yes': 1, 'No': 0}
        property_area_map = {'Urban': 2, 'Rural': 0, 'Semiurban': 1}

        input_df['Gender'] = gender_map[input_data['Gender']]
        input_df['Married'] = married_map[input_data['Married']]
        input_df['Dependents'] = dependents_map[input_data['Dependents']]
        input_df['Education'] = education_map[input_data['Education']]
        input_df['Self_Employed'] = self_employed_map[input_data['Self_Employed']]
        input_df['Property_Area'] = property_area_map[input_data['Property_Area']]
        # for col in cols:
        #     input_df[col] = le.fit_transform(input_df[col])

        prediction = model.predict(input_df)
        return prediction
    
    if st.button('Predict'):
        prediction = predict_loan(input_data)
        if prediction == 'Y':
            st.success('Congrats you are eligible for a loan')
        else:
            st.error('Sorry you did not meet the requirement.')
        
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap.initjs()
        st.subheader("SHAP Explanation")

        # Use SHAP waterfall plot for class 1 (if binary classification)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0][1],
            base_values=explainer.expected_value[1],
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ), max_display=10, show=False)

        st.pyplot(fig)
        plt.clf()

