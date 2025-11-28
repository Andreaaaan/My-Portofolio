"""
This module contains a Streamlit app for showcasing a portfolio and analyzing data.
It includes functions for loading CSS, building UI components, and processing data.

Author: Muhammad Andrean
Date: January 2025
"""

import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load CSS file
def load_css(css_file):
    """
    Load a CSS file and apply its styles to the Streamlit app.

    Args:
        css_file (str): Path to the CSS file to be loaded.
    """
    with open(css_file, encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load CSS
load_css("styles.css")

# Sidebar
page = st.sidebar.selectbox(
    "Menu", 
    [
        "About Me", 
        "Customer Segmentation with XGboost and Random Forest", 
        "Customer Segmentation With Gradient Boosting", 
        "Customer Segmentation Using Unsupervised Learning"
        "Prediction Test Covid-19 using Machine learning"
    ]
)

# Switch Page based on selection
if page == "About Me":
    st.title("Portfolio")
    col1, col2 = st.columns(
    [1, 2]
)

    with col1:
        st.image("profile_photo.jpg", use_column_width=False)

    with col2:
        st.header("Muhammad Andrean")
        st.write("""
        <div class="contact-info">
            <p>
                <strong></strong> 
                <a class="email-link" href="mailto:muhammadandrean4514@gmail.com" target="_blank">
                    ✉️ muhammadandrean4514@gmail.com
                </a>
            </p>
            <p>
                <strong></strong> 
                <a class="linkedin-link" href="https://www.linkedin.com/in/muhandrean/" target="_blank">
                    🔗 LinkedIn Profile
                </a>
            </p>
            <p>
                <strong></strong> 
                <a class="github-link" href="https://github.com/Andreaaaan/" target="_blank">
                    🐙 GitHub Repositories
                </a>
            </p>
            <p>
                <strong></strong> 
                <a class="portfolio-link" href="https://muhammad-andrean-portfolio.streamlit.app/" target="_blank">
                    🌐 View My Portfolio
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("Summary")
    st.write("""
    As a driven and enthusiastic student of Informatics Engineering, I'm passionate about exploring the vast potential of data analysis and data science. With a strong foundation in computer systems and programming, I'm excited to dive deeper into the world of data-driven insights and decision-making.

    Beyond my technical skills, I thrive in collaborative environments where teamwork and open communication are valued. I believe that diverse perspectives and collective efforts can lead to innovative solutions and meaningful outcomes.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("Top Skills")
    skills = ["""
            - Python
            - C (Programming Language) 
            - R (Programming Language)
            - SQL
            """]
    st.write(", ".join(skills))

    st.markdown("<hr>", unsafe_allow_html=True)

    # Languages
    st.header("Languages")
    languages = ["""
                 - English (Professional Working Proficiency)
                 - Bahasa Indonesia (Native)
                 """]
    st.write(", ".join(languages))

    st.markdown("<hr>", unsafe_allow_html=True)

    # Certifications
    st.header("Certifications")
    certifications = [
        """
        - BNSP Certified Associate Data Science
        - Certified Data Scientist by Digital Skola
        - Data Science Fundamental Certifitaction by DQLab
        """
    ]
    st.write(", ".join(certifications))

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Education
    st.header("Education")
    education = [
            """
            - Informatics Student at Universitas Multimedia Nusantara
            - Graduated Data Science Bootcamp student at Digital SKola
            """
        ] 
    st.write(", ".join(education))

# Customer Segmentation with XGBoost and random forest
elif page == "Customer Segmentation with XGboost and Random Forest":
    st.title("Aplikasi Prediksi Customer Segmentation dengan Random Forest dan XGBoost")

    # Load the trained models XGBoost and Random Forest
    rf_model = joblib.load('random_forest_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')

    # Label Encoders for categorical features (XGBoost and Random Forest)
    gender_encoder = LabelEncoder()
    gender_encoder.classes_ = np.array(['Female', 'Male'])

    married_encoder = LabelEncoder()
    married_encoder.classes_ = np.array(['No', 'Yes'])

    graduated_encoder = LabelEncoder()
    graduated_encoder.classes_ = np.array(['No', 'Yes'])

    profession_encoder = LabelEncoder()
    profession_encoder.classes_ = np.array(['Artist',
                                            'Doctor',
                                            'Engineer',
                                            'Entertainment',
                                            'Healthcare',
                                            'Lawyer'])

    spending_score_encoder = LabelEncoder()
    spending_score_encoder.classes_ = np.array(['Low', 'Average', 'High'])

    var_1_encoder = LabelEncoder()
    var_1_encoder.classes_ = np.array(['Cat_1',
                                       'Cat_2',
                                       'Cat_3',
                                       'Cat_4',
                                       'Cat_5',
                                       'Cat_6',
                                       'Cat_7'])

    # User Inputs (XGBoost and Random Forest)
    gender = st.selectbox("Jenis Kelamin", ('Male', 'Female'))
    ever_married = st.selectbox("Pernah Menikah", ('Yes', 'No'))
    age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    graduated = st.selectbox("Lulusan Universitas", ('Yes', 'No'))
    profession = st.selectbox("Profesi", ('Artist', 'Doctor',
                                          'Engineer', 'Entertainment',
                                          'Healthcare', 'Lawyer'))
    work_experience = st.number_input("Pengalaman Kerja (Tahun)",
                                      min_value=0, max_value=40, value=5)
    spending_score = st.selectbox("Spending Score", ('Low', 'Average', 'High'))
    family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
    var_1 = st.selectbox("Kategori Var_1", ('Cat_1',
                                            'Cat_2',
                                            'Cat_3',
                                            'Cat_4',
                                            'Cat_5',
                                            'Cat_6',
                                            'Cat_7'))

    # Transform input categorical Into numerical
    gender_encoded = gender_encoder.transform([gender])[0]
    married_encoded = married_encoder.transform([ever_married])[0]
    graduated_encoded = graduated_encoder.transform([graduated])[0]
    profession_encoded = profession_encoder.transform([profession])[0]
    spending_score_encoded = spending_score_encoder.transform([spending_score])[0]
    var_1_encoded = var_1_encoder.transform([var_1])[0]

    # features input as an array
    features = [gender_encoded,
                married_encoded,
                age,
                graduated_encoded,
                profession_encoded,
                work_experience,
                spending_score_encoded,
                family_size,
                var_1_encoded]

    # Function to get customer segment description
    def get_customer_segment_description(segment_id):
        if segment_id == 0:
            return "Segmen 0: Pelanggan Premium"
        elif segment_id == 1:
            return "Segmen 1: Pelanggan Setia"
        elif segment_id == 2:
            return "Segmen 2: Pelanggan Potensial"
        elif segment_id == 3:
            return "Segmen 3: Pelanggan Hemat"
        else:
            return "Segmen tidak dikenal."


    # Random Forest Prediction
    if st.button('Prediksi Random Forest'):
        prediction_rf = rf_model.predict([features])[0]
        st.write(f'Prediksi Random Forest: {prediction_rf}')
        st.write(get_customer_segment_description(prediction_rf))

    # XGboost prediction
    if st.button('Prediksi XGBoost'):
        prediction_xgb = xgb_model.predict([features])[0]
        st.write(f'Prediksi XGBoost: {prediction_xgb}')
        st.write(get_customer_segment_description(prediction_xgb))

# Customer Segmentation with Unsupervised Learning (KMeans)    
elif page == "Customer Segmentation Using Unsupervised Learning":
    st.title("Customer Segmentation Using Unsupervised Learning (KMeans)")

    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')

    st.title("Customer Segmentation Using Unsupervised Learning (KMeans)")

    # User Inputs for KMeans model
    age = st.slider("Age", 18, 100, 30)
    work_experience = st.slider("Work Experience (Years)", 0, 40, 5)
    family_size = st.slider("Family Size", 1, 10, 3)
    gender = st.selectbox("Gender", ("Male", "Female"))
    ever_married = st.selectbox("Ever Married", ("Yes", "No"))
    spending_score = st.selectbox("Spending Score", ("Low", "High"))

    # Encode categorical features
    gender_encoded = 1 if gender == "Male" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    spending_score_encoded = 1 if spending_score == "High" else 0

    # Combine all features into a single array
    features = [age,
                work_experience,
                family_size,
                gender_encoded,
                ever_married_encoded,
                spending_score_encoded]
    input_data = np.array(features).reshape(1, -1)

    # Standardize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the cluster using the KMeans model
    cluster = kmeans_model.predict(input_data_scaled)[0]

    # Map the cluster to segment labels (A, B, C, D)
    cluster_to_segment = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    segment = cluster_to_segment.get(cluster, "Unknown")

    # Display the segment prediction
    st.write(f"The predicted customer segment is: **Segment {segment}**")

    # Segment descriptions   
    def get_segment_description(segment_label):
        descriptions = {
            'A': "Segment A: Price-sensitive customers who are responsive to promotions and discounts.",
            'B': "Segment B: Potential customers who show interest but don't purchase frequently.",
            'C': "Segment C: Loyal customers who make regular purchases.",
            'D': "Segment D: High-value customers who prioritize quality and premium services."
        }
        return descriptions.get(segment_label, "Unknown segment")

    # Display the segment description
    st.write(get_segment_description(segment))

# Customer Segmentation with Gradient Boosting
elif page == "Customer Segmentation With Gradient Boosting":
    st.title("Customer Segmentation with Gradient Boosting")

    # Load the Gradient Boosting model
    gb_model = joblib.load('gradient_boosting_model.pkl')

    # Define Label Encoders for categorical features
    gender_encoder = LabelEncoder()
    gender_encoder.classes_ = np.array(['Female', 'Male'])

    married_encoder = LabelEncoder()
    married_encoder.classes_ = np.array(['No', 'Yes'])

    graduated_encoder = LabelEncoder()
    graduated_encoder.classes_ = np.array(['No', 'Yes'])

    profession_encoder = LabelEncoder()
    profession_encoder.classes_ = np.array(['Artist',
                                            'Doctor',
                                            'Engineer',
                                            'Entertainment',
                                            'Healthcare',
                                            'Lawyer'])

    spending_score_encoder = LabelEncoder()
    spending_score_encoder.classes_ = np.array(['Low', 'Average', 'High'])

    var_1_encoder = LabelEncoder()
    var_1_encoder.classes_ = np.array(['Cat_1',
                                       'Cat_2',
                                       'Cat_3',
                                       'Cat_4',
                                       'Cat_5',
                                       'Cat_6',
                                       'Cat_7'])

    # Users inputs for Gradient Boosting model
    gender = st.selectbox("Jenis Kelamin", ('Male', 'Female'))
    ever_married = st.selectbox("Pernah Menikah", ('Yes', 'No'))
    age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    graduated = st.selectbox("Lulusan Universitas", ('Yes', 'No'))
    profession = st.selectbox("Profesi", ('Artist',
                                          'Doctor',
                                          'Engineer',
                                          'Entertainment',
                                          'Healthcare',
                                          'Lawyer'))
    work_experience = st.number_input("Pengalaman Kerja (Tahun)",
                                      min_value=0, max_value=40, value=5)
    spending_score = st.selectbox("Spending Score", ('Low', 'Average', 'High'))
    family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
    var_1 = st.selectbox("Kategori Var_1", ('Cat_1',
                                            'Cat_2',
                                            'Cat_3',
                                            'Cat_4',
                                            'Cat_5',
                                            'Cat_6',
                                            'Cat_7'))

    # Encode inputs for Gradient Boosting model
    gender_encoded = gender_encoder.transform([gender])[0]
    married_encoded = married_encoder.transform([ever_married])[0]
    graduated_encoded = graduated_encoder.transform([graduated])[0]
    profession_encoded = profession_encoder.transform([profession])[0]
    spending_score_encoded = spending_score_encoder.transform([spending_score])[0]
    var_1_encoded = var_1_encoder.transform([var_1])[0]

    # Combine all features into an array for Gradient Boosting model
    features = [gender_encoded,
                married_encoded,
                age,
                graduated_encoded,
                profession_encoded,
                work_experience,
                spending_score_encoded,
                family_size,
                var_1_encoded]

   # Function to get customer segment description
    def get_customer_segment_description(segment_code):
        if segment_code == 0:
            return "Segmen A: Pelanggan Premium - Pelanggan bernilai tinggi yang lebih mengutamakan kualitas dan layanan premium."
        elif segment_code == 1:
            return "Segmen B: Pelanggan Setia - Pelanggan yang sudah sering melakukan pembelian dan loyal."
        elif segment_code == 2:
            return "Segmen C: Pelanggan Potensial - Pelanggan baru yang menunjukkan minat namun belum sering membeli."
        elif segment_code == 3:
            return "Segmen D: Pelanggan Hemat - Pelanggan yang sangat peka terhadap harga dan promosi."
        else:
            return "Segmen tidak dikenal."

    # Gradient Boosting Prediction
    if st.button('Prediksi dengan Gradient Boosting'):
        prediction_gb = gb_model.predict([features])[0]
        st.write(f'Prediksi Gradient Boosting: {prediction_gb}')
        st.write(get_customer_segment_description(prediction_gb))


# Prediction Test Covid-19 using Machine learning
elif page == "Prediction Test Covid-19 using Machine learning":
    st.title("COVID-19 Prediction Test using Machine Learning")
    
    @st.cache
    def load_model():
        return joblib.load('model_parallel.pkl')
    
    model = load_model()
    
    label_encoder = LabelEncoder()
    label_encoder.fit(['Yes', 'No'])
    
    st.subheader("Jawab pertanyaan berikut dengan memilih Yes atau No")
    
    questions = [
        "Breathing Problem","Fever","Dry Cough","Sore Throat","Running Nose",
        "Asthma","Chronic Lung Disease","Headache","Heart Disease","Diabetes",
        "Fatigue","Gastrointestinal","abroad travel","Contact with COVID Patient",
        "Attended Large Gathering","Visited Public Exposed Places",
        "Family working in Public Exposed Places","Wearing Masks",
        "Sanitization from Market","pernah kontak dengan pasien covid19"
    ]
    
    user_input = []
    for question in questions:
        # Display the question and Yes/No radio button in one row
        col1, col2 = st.columns([3, 1]) 
        with col1:
            st.write(f"**{question}**")
        with col2:
            answer = st.radio("", ['Yes', 'No'], key=question, horizontal=True)
            encoded_answer = label_encoder.transform([answer])[0]
            user_input.append(encoded_answer)
            
        st.markdown("<hr>", unsafe_allow_html=True)


    if st.button("Predict COVID-19"):
        try:
            prediction = model.predict([user_input])
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            st.success(f"The prediction result is: **{result}**")
        except ValueError as e:
            st.error(f"Error: {e}")
            st.write("Pastikan jumlah fitur input sesuai dengan model yang digunakan.")
