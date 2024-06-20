import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Memuat data Big Mart
@st.cache_data
def load_data():
    return pd.read_csv('Train.csv')

big_mart_data = load_data()

# Judul aplikasi
st.title('Prediksi Penjualan Supermarket')

# Menampilkan dataset
if st.checkbox('Tampilkan data'):
    st.subheader('Data Penjualan Supermarket')
    st.write(big_mart_data)

# Praproses data
# Mengisi nilai yang hilang di Item_Weight
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# Mengisi nilai yang hilang di Outlet_Size menggunakan modus berdasarkan Outlet_Type
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values, 'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# Mengganti kategori 'Item_Fat_Content'
big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

# Encoding variabel kategori
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

# Membagi data menjadi set pelatihan dan pengujian
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Pelatihan model XGBoost
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Prediksi
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)

# Menampilkan metrik evaluasi
st.subheader('Evaluasi Model')
st.write('R Squared (R2) skor pada data pelatihan:', r2_train)
#st.write('R Squared (R2) skor pada data pengujian:', r2_test)

# Visualisasi tambahan (opsional)
st.subheader('Visualisasi Data')

# Plot distribusi untuk Item_Weight
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Weight'], kde=True)
st.pyplot(plt.gcf())

# Plot distribusi untuk Item_Visibility
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_Visibility'], kde=True)
st.pyplot(plt.gcf())

# Plot distribusi untuk Item_MRP
plt.figure(figsize=(6, 6))
sns.histplot(big_mart_data['Item_MRP'], kde=True)
st.pyplot(plt.gcf())

# Plot jumlah untuk Outlet_Establishment_Year
plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
st.pyplot(plt.gcf())

# Plot jumlah untuk Item_Fat_Content
plt.figure(figsize=(6, 6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
st.pyplot(plt.gcf())

# Plot jumlah untuk Item_Type (gambar besar untuk menampilkan semua kategori)
plt.figure(figsize=(30, 6))
sns.countplot(x='Item_Type', data=big_mart_data)
st.pyplot(plt.gcf())

# Plot jumlah untuk Outlet_Size
plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
st.pyplot(plt.gcf())

# Input prediksi
st.subheader('Prediksi Jumlah Penjualan')
Item_Identifier = st.number_input('ID Item', min_value=0, max_value=big_mart_data['Item_Identifier'].max())
Item_Weight = st.number_input('Berat Item', min_value=0.0, max_value=big_mart_data['Item_Weight'].max(), value=12.0)
Item_Fat_Content = st.selectbox('Kandungan Lemak Item', big_mart_data['Item_Fat_Content'].unique())
Item_Visibility = st.number_input('Visibilitas Item', min_value=0.0, max_value=big_mart_data['Item_Visibility'].max(), value=0.05)
Item_Type = st.selectbox('Tipe Item', big_mart_data['Item_Type'].unique())
Item_MRP = st.number_input('MRP Item', min_value=0.0, max_value=big_mart_data['Item_MRP'].max(), value=150.0)
Outlet_Identifier = st.selectbox('ID Outlet', big_mart_data['Outlet_Identifier'].unique())
Outlet_Establishment_Year = st.number_input('Tahun Pendirian Outlet', min_value=1985, max_value=big_mart_data['Outlet_Establishment_Year'].max())
Outlet_Size = st.selectbox('Ukuran Outlet', big_mart_data['Outlet_Size'].unique())
Outlet_Location_Type = st.selectbox('Tipe Lokasi Outlet', big_mart_data['Outlet_Location_Type'].unique())
Outlet_Type = st.selectbox('Tipe Outlet', big_mart_data['Outlet_Type'].unique())

input_data = (Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP,
              Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Memprediksi penjualan
prediction = regressor.predict(input_data_reshaped)
st.write(f'Prediksi Jumlah Pendapatan Penjualan Item dalam USD: {prediction[0]:.2f}')
