import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load the Big Mart data
@st.cache_data
def load_data():
    return pd.read_csv('kecerdasanBuatanUASPush\Train.csv')

big_mart_data = load_data()

# Title of the application
st.title('Big Mart Sales Prediction')

# Display the dataset
if st.checkbox('Show raw data'):
    st.subheader('Big Mart Sales Data')
    st.write(big_mart_data)

# Data preprocessing
# Fill missing values in Item_Weight
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# Fill missing values in Outlet_Size using mode based on Outlet_Type
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull() 
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# Replace 'Item_Fat_Content' categories
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

# Splitting the data into training and testing sets
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# XGBoost model training
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Predictions
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)

# Display evaluation metrics
st.subheader('Model Evaluation')
st.write('R Squared (R2) score on training data:', r2_train)

# Additional visualizations (optional)
st.subheader('Data Visualization')

# Distribution plot for Item_Weight
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Weight'], kde=True)
st.pyplot(plt.gcf())

# Distribution plot for Item_Visibility
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
st.pyplot(plt.gcf())

# Distribution plot for Item_MRP
plt.figure(figsize=(6,6))
sns.histplot(big_mart_data['Item_MRP'], kde=True)
st.pyplot(plt.gcf())

# Count plot for Outlet_Establishment_Year
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
st.pyplot(plt.gcf())

# Count plot for Item_Fat_Content
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
st.pyplot(plt.gcf())

# Count plot for Item_Type (large figure to show all categories)
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=big_mart_data)
st.pyplot(plt.gcf())

# Count plot for Outlet_Size
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
st.pyplot(plt.gcf())

# Prediction input
st.subheader('Prediksi Jumlah Penjualan')

# Sorting the unique values
sorted_Item_Identifier = sorted(big_mart_data['Item_Identifier'].unique())
sorted_Item_Fat_Content = sorted(big_mart_data['Item_Fat_Content'].unique())
sorted_Item_Type = sorted(big_mart_data['Item_Type'].unique())
sorted_Outlet_Identifier = sorted(big_mart_data['Outlet_Identifier'].unique())
sorted_Outlet_Size = sorted(big_mart_data['Outlet_Size'].unique())
sorted_Outlet_Location_Type = sorted(big_mart_data['Outlet_Location_Type'].unique())
sorted_Outlet_Type = sorted(big_mart_data['Outlet_Type'].unique())

Item_Identifier = st.selectbox('Item Identifier', sorted_Item_Identifier)
Item_Weight = st.number_input('Item Weight', min_value=0.0, max_value=big_mart_data['Item_Weight'].max(), value=12.0)
Item_Fat_Content = st.selectbox('Item Fat Content', sorted_Item_Fat_Content)
Item_Visibility = st.number_input('Item Visibility', min_value=0.0, max_value=big_mart_data['Item_Visibility'].max(), value=0.05)
Item_Type = st.selectbox('Item Type', sorted_Item_Type)
Item_MRP = st.number_input('Item MRP', min_value=0.0, max_value=big_mart_data['Item_MRP'].max(), value=150.0)
Outlet_Identifier = st.selectbox('Outlet Identifier', sorted_Outlet_Identifier)
Outlet_Establishment_Year = st.number_input('Outlet Establishment Year', min_value=1985, max_value=big_mart_data['Outlet_Establishment_Year'].max())
Outlet_Size = st.selectbox('Outlet Size', sorted_Outlet_Size)
Outlet_Location_Type = st.selectbox('Outlet Location Type', sorted_Outlet_Location_Type)
Outlet_Type = st.selectbox('Outlet Type', sorted_Outlet_Type)

input_data = (Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP,
              Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predicting sales
prediction = regressor.predict(input_data_reshaped)
st.write(f'Prediksi Jumlah Penjualan dalam USD: {prediction[0]:.2f}')
