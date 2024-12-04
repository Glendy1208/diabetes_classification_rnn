import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model RNN yang telah dilatih
model = load_model('diabetes_rnn.h5')

# Fungsi untuk normalisasi input (hanya fitur yang relevan: 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level')
def normalize_inputs(inputs, scaler):
    # Hanya normalisasi kolom yang relevan (kolom 2, 5, 6, 7: 'age', 'bmi', 'HbA1c_level', 'blood_glucose_level')
    inputs_to_normalize = inputs[[1, 5, 6, 7]]  # Ambil hanya age, bmi, HbA1c_level, blood_glucose_level
    scaled = scaler.transform([inputs_to_normalize])  # Normalisasi data
    inputs[[1, 5, 6, 7]] = scaled[0]  # Kembalikan hasil normalisasi ke posisi yang benar dalam array input
    return inputs

# Main function untuk Streamlit app
def main():
    # Sidebar menu
    st.sidebar.title('Klasifikasi Diabetes dengan RNN')
    menu = st.sidebar.radio("Menu", 
                            options=["Data Understanding", "Preprocessing", "Modelling", "Informasi Inputan", "Form Inputan"])

    if menu == "Data Understanding":
        st.title("Data Understanding")
        st.header("Informasi Data")
        st.subheader("Diabetes Dataset Original")
        data = pd.read_csv('csv/dia_ori_csv.csv')
        st.write(data)
        st.subheader("Jumlah Kelas pada Kolom Diabetes")
        st.write(data['diabetes'].value_counts())
        st.image('img/target-num.png', caption='Distribusi Fitur')
        st.subheader("Korelasi Fitur dengan Target")
        st.write("Berikut adalah distribusi dari beberapa fitur yang relevan dalam dataset:")
        st.image('img/corr.png', caption='Korelasi Fitur dengan Target')
        st.subheader("Informasi Fitur dan Target")
        st.write("""
            Dataset ini terdiri dari 8 fitur dan 1 target (label) yang digunakan untuk memprediksi apakah seseorang terkena diabetes:

            **Fitur :**
            - **Gender**:
                - Female : pasien berjenis kelamin perempuan
                - Male : pasien berjenis kelamin laki-laki
                - Other : jenis kelamin lainnya
            - **Age**: Usia pasien
            - **Hypertension**:
                - 0 (Tidak) : pasien tidak memiliki hipertensi
                - 1 (Ya) : pasien memiliki hipertensi
            - **Heart Disease**: 
                - 0 (Tidak) : pasien tidak memiliki penyakit jantung
                - 1 (Ya) : pasien memiliki penyakit jantung
            - **Smoking History**: 
                - current : pasien merokok saat ini
                - ever : pasien pernah merokok
                - former : pasien mantan perokok
                - never : pasien tidak pernah merokok
                - not current : pasien tidak merokok saat ini
                - No Info : informasi merokok tidak tersedia (missing value)
            - **BMI**: Indeks Massa Tubuh
            - **HbA1c Level**: Level HbA1c dalam darah, diambil dari rata-rata HbA1c dalam 3 bulan terakhir.
            - **Blood Glucose Level**: Level glukosa dalam darah, diambil pada saat pemeriksaan saat ini.

            **Target :**
            - **Diabetes**:
                - 0 (tidak) : pasien tidak terkena diabetes
                - 1 (ya) : pasien terkena diabetes
        """)

    elif menu == "Preprocessing":
        st.title("Preprocessing data")
        st.write("Proses preprocessing data dilakukan untuk mempersiapkan data sebelum dimasukkan ke dalam model RNN. Proses ini berfungsi untuk meningkatkan kualitas dari data sehingga menghasilkan model yang lebih baik.")
        st.subheader("1. Encoding Kategorikal Data")
        st.write("pada dataset terdapat beberapa fitur yang masih berbentuk kategorikal string, sehingga perlu diubah menjadi bentuk kategorikal numerik agar dapat diolah oleh model. Berikut adalah mapping yang digunakan:")
        st.write("""
            1. Gender
                 - Perempuan : 0
                 - Laki-laki : 1
                 - Other : 2
            2. Smoking History
                 - current : 0
                 - ever : 1
                 - former : 2
                 - never : 3
                 - not current : 4
        """)
        st.subheader("2. Imputasi Missing Value")
        st.write("Pada kolom `smoking_history` terdapat beberapa nilai yang hilang yaitu `No Info`. Untuk mengatasi hal ini, digunakan teknik KNN Imputer untuk mengisi nilai-nilai yang hilang dengan nilai K=5.")
        st.subheader("3. Normalisasi Data")
        st.write("Fitur `age`, `bmi`, `HbA1c_level`, dan `blood_glucose_level` adalah fitur yang bertipe numerik. Oleh karena itu perlu dinormalisasi agar memiliki skala yang sama. Normalisasi dilakukan menggunakan MinMaxScaler.")
        st.subheader("4. Splitting Data")
        st.write("Data dibagi menjadi data latih dan data uji dengan perbandingan 80:20. Data latih digunakan untuk melatih model, sedangkan data uji digunakan untuk menguji performa model.")
        st.subheader("5. Resampling Data Training")
        st.write("Karena dilihat dari distribusi kelas yang tidak seimbang, maka perlu dilakukan resampling data training supaya model tidak cenderung memprediksi kelas mayoritas saja. Teknik resampling yang digunakan adalah ADASYN dan akan dijadikan distribuksi untuk kelas 0 (tidak terkena diabetes) menjadi 20000 record dan kelas 1 (terkena diabetes) menjadi 8745 record.")
        st.header("Hasil Preprocessing")
        train_df = pd.read_csv("csv/train_diabetes.csv")
        test_df = pd.read_csv("csv/test_diabetes.csv")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Train")
            st.write(train_df)
            st.write("**Distribusi Kelas**")
            st.write(train_df['diabetes'].value_counts())
            st.write("**jumlah data dan fitur**")
            st.write(train_df.shape)

        with col2:
            st.subheader("Data Test")
            st.write(test_df)
            st.write("**Distribusi Kelas**")
            st.write(test_df['diabetes'].value_counts())
            st.write("**jumlah data dan fitur**")
            st.write(test_df.shape)

    elif menu == "Modelling":
        st.title("Modelling dengan Recurrent Neural Network (RNN)")
        st.write("Setelah data telah bersih dan siap, maka tahap selanjutnya adalah membangun model machine learning untuk memprediksi diabetes. Model yang digunakan adalah Recurrent Neural Network (RNN) yang merupakan jenis arsitektur deep learning yang cocok untuk data sekuensial.")
        st.header("Arsitektur Model")
        st.write("Berikut adalah arsitektur model RNN yang digunakan untuk memprediksi diabetes:")
        st.image('img/arsitektur.png', caption='Arsitektur Model RNN')
        st.write("""
            1. **Input Layer**:
                - Model menerima input dari dataset yang terdiri dari **8 fitur** untuk setiap sampel. 
                - Dalam konteks RNN, ini berarti bahwa data input terdiri dari urutan (sequence) di mana setiap langkah waktu (timestep) berisi 8 fitur.
                
            2. **RNN Layer**:
                - Lapisan **SimpleRNN** dengan **64 unit** digunakan untuk memproses data urutan (sequence data). Setiap unit di dalam RNN berfungsi untuk mempelajari pola temporal dalam data.
                - Fungsi aktivasi yang digunakan adalah **ReLU** (Rectified Linear Unit), yang sering digunakan untuk mencegah masalah vanishing gradient dan memberikan performa yang baik dalam pelatihan.
            
            3. **Dropout Layer**:
                - Untuk mengurangi kemungkinan **overfitting**, lapisan **Dropout** dengan rate 30% (`Dropout(0.3)`) ditambahkan setelah lapisan RNN. Dropout secara acak menonaktifkan beberapa neuron selama pelatihan untuk meningkatkan generalisasi model.
                
            4. **Dense Layer**:
                - Setelah RNN, model menggunakan lapisan **Dense** dengan **32 unit** dan fungsi aktivasi **ReLU**. Lapisan ini bertanggung jawab untuk memproses fitur yang telah diekstraksi oleh RNN.
                - **Regularisasi L2** diterapkan pada lapisan ini untuk mengurangi kemungkinan overfitting dengan menambahkan penalti terhadap bobot yang besar.
            
            5. **Batch Normalization**:
                - **BatchNormalization** diterapkan setelah lapisan Dense untuk mempercepat konvergensi dan stabilitas model, serta mengurangi sensitivitas terhadap perubahan distribusi data selama pelatihan.
            
            6. **Output Layer**:
                - Lapisan output terdiri dari **1 unit** dengan aktivasi **sigmoid**, yang sangat cocok untuk masalah **klasifikasi biner**. Aktivasi sigmoid mengubah output menjadi nilai antara 0 dan 1, yang dapat dipetakan ke dua kelas: **diabetes** atau **tidak diabetes**.
                
            7. **Optimizer**:
                - **Adam optimizer** digunakan untuk mempercepat pelatihan dan memberikan konvergensi yang lebih baik. Adam adalah algoritma optimisasi adaptif yang menggabungkan keunggulan dari dua algoritma optimisasi yang lebih lama (AdaGrad dan RMSProp).
                - Fungsi loss yang digunakan adalah **binary crossentropy**, yang merupakan pilihan standar untuk masalah klasifikasi biner seperti ini.
                
            8. **EarlyStopping**:
                - Untuk mencegah **overfitting** lebih lanjut, **early stopping** digunakan. Early stopping menghentikan pelatihan jika model tidak menunjukkan perbaikan pada data validasi dalam sejumlah epoch berturut-turut. Ini membantu untuk menjaga model tetap sederhana dan tidak terlalu menyesuaikan diri dengan data pelatihan.
        """)
        st.header("Evaluasi Model")
        st.image('img/acc-loss.png', caption='Grafik Akurasi dan Loss Model')
        st.image('img/confuse.png', caption='Grafik Confusion Matrix Model')
        st.image('img/metric_eval.png', caption='Metric Evaluasi Model')

    elif menu == "Informasi Inputan":
        st.title("Informasi Inputan")
        st.write("""
            Berikut adalah deskripsi mengenai inputan yang diperlukan untuk melakukan prediksi:
            - **Gender**:
                - Female : pasien berjenis kelamin perempuan
                - Male : pasien berjenis kelamin laki-laki
                - Other : jenis kelamin lainnya
            - **Age**: inputan untuk usia pasien dalam tahun
            - **Hypertension**:
                - No : pasien tidak memiliki hipertensi
                - Yes : pasien memiliki hipertensi
            - **Heart Disease**: 
                - No : pasien tidak memiliki penyakit jantung
                - Yes : pasien memiliki penyakit jantung
            - **Smoking History**: 
                - current : pasien merokok saat ini
                - ever : pasien pernah merokok
                - former : pasien mantan perokok
                - never : pasien tidak pernah merokok
                - not current : pasien tidak merokok saat ini
            - **BMI**: Inputan Indeks Massa Tubuh pasien
            - **HbA1c Level**: Level HbA1c dalam darah pasien, diambil dari rata-rata gula darah dalam 3 bulan terakhir.
            - **Blood Glucose Level**: Level glukosa dalam darah, diambil pada saat pemeriksaan saat ini.
        """)

    elif menu == "Form Inputan":
        st.title("Form Inputan")
        st.write("Masukkan informasi pasien di bawah ini untuk prediksi diabetes:")

        # Form input dari pengguna
        gender = st.selectbox("Gender", options=["Female", "Male", "Other"], index=0)
        age = st.number_input("Age (in years)", min_value=1, max_value=120, value=30, step=1)
        hypertension = st.selectbox("Hypertension", options=["No", "Yes"], index=0)
        heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"], index=0)
        smoking_history = st.selectbox(
            "Smoking History", 
            options=["current", "ever", "former", "never", "not current"], 
            index=3
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)  # updated to support float
        hba1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100, step=1)

        # Mapping untuk konversi input ke nilai numerik
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        hypertension_map = {'No': 0, 'Yes': 1}
        heart_disease_map = {'No': 0, 'Yes': 1}
        smoking_history_map = {
            'current': 0,
            'ever': 1,
            'former': 2,
            'never': 3, 
            'not current': 4
        }

        # Convert inputs ke nilai numerik
        gender = gender_map[gender]
        hypertension = hypertension_map[hypertension]
        heart_disease = heart_disease_map[heart_disease]
        smoking_history = smoking_history_map[smoking_history]

        # Membuat array input untuk fitur
        input_features = np.array([gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level])

        # Normalisasi
        scaler = MinMaxScaler(feature_range=(0, 1))
        min_max_values = np.array([
            [0.08, 10.01, 3.5, 80],      # Minimum values
            [80.0, 95.69, 9.0, 300]      # Maximum values
        ])
        scaler.fit(min_max_values)  # Fit scaler dengan min dan max yang baru

        # Normalisasi input (hanya fitur yang relevan)
        normalized_input = normalize_inputs(input_features, scaler)

        # Mengubah bentuk data input untuk RNN
        normalized_input = normalized_input.reshape((1, normalized_input.shape[0], 1))

        # Menunggu prediksi ketika tombol ditekan
        if st.button("Predict"):
            prediction = model.predict(normalized_input)
            prediction = (prediction > 0.5).astype(int)  # Threshold = 0.5

            if prediction[0][0] == 1:
                st.error("Pasien **terkena diabetes**.")
            else:
                st.success("Pasien **tidak terkena diabetes**.")

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
