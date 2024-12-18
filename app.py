import pandas as pd
import pickle
import streamlit as st

# Load model dan vectorizer
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_vectorizer = open('tfidf_vectorizer.pkl', 'rb')
vectorizer = pickle.load(pickle_vectorizer)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('/content/streamlit.csv')  # Ganti dengan file Anda
    return df

df = load_data()

# Fungsi prediksi
def prediction(title, company, location, salary):
    # Gabungkan input dalam satu string atau array
    user_input = f"{title} {company} {location} {salary}"

    # Transform input menggunakan TF-IDF
    input_vectorized = vectorizer.transform([user_input])

    # Prediksi menggunakan model
    prediction = classifier.predict(input_vectorized)

    # Kembalikan hasil prediksi
    job_map = {0: 'Tidak Layak', 1: 'Layak'}
    job_name = job_map[prediction[0]]
    return job_name

# Main app
def main():
    st.title('Prediksi Kelayakan Kandidat Kerja')

    # Membuat dropdown dengan variasi unik dari data
    title_options = df['Title'].unique()
    company_options = df['Company'].unique()
    location_options = df['Location'].unique()
    salary_options = df['Salary'].unique()

    # Dropdown untuk input
    title = st.selectbox('Pilih Title', title_options)
    company = st.selectbox('Pilih Company', company_options)
    location = st.selectbox('Pilih Location', location_options)
    salary = st.selectbox('Pilih Salary', salary_options)

    result = ""

    if st.button('Prediksi'):
        result = prediction(title, company, location, salary)
    st.success(result)

if __name__ == '__main__':
    main()
