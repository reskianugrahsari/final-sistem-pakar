from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

dataset_path = 'data_DBD.csv'
df = pd.read_csv(dataset_path)

# Menghapus kolom yang tidak diperlukan
df = df.drop(['Umur', 'Jenis Kelamin', 'Gejalah', 'Durasi Gejalah', 'Riwayat Kesehatan', 'Pemeriksaan Fisik',
              'Pemeriksaan Laboratorium', 'Genetik', 'Faktor Resiko Lainnya', 'Data Komorbidatas', 'Kronologi Gejalah'],
             axis=1)

le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# One-hot encoding selama pelatihan
X_train_encoded = pd.get_dummies(X)

# Simpan nama kolom yang dihasilkan saat pelatihan
train_columns = X_train_encoded.columns

X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, y, test_size=0.2, random_state=42)

# Inisialisasi model K-NN
default_k = 3
knn_model = KNeighborsClassifier(n_neighbors=default_k)
knn_model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        umur = float(request.form['umur'])
        jenis_kelamin = request.form['jenis_kelamin']
        jumlah_gejala = int(request.form['jumlah_gejala'])

        # One-hot encoding selama prediksi
        input_data = pd.DataFrame({'Umur': [umur], 'Jenis Kelamin': [jenis_kelamin]})
        input_data_encoded = pd.get_dummies(input_data)

        # Memastikan bahwa kolom-kolom yang muncul selama prediksi sama dengan yang ada selama pelatihan
        missing_cols = set(train_columns) - set(input_data_encoded.columns)
        for col in missing_cols:
            input_data_encoded[col] = 0

        # Pastikan urutan kolom sama dengan yang digunakan saat pelatihan
        input_data_encoded = input_data_encoded[train_columns]

        # Fit ulang model K-NN dengan nilai k yang baru
        knn_model = KNeighborsClassifier(n_neighbors=jumlah_gejala)
        knn_model.fit(X_train, y_train)

        prediction = knn_model.predict(input_data_encoded)

        # Mengganti nilai hasil prediksi dengan nilai asli dari kolom "Diagnosis"
        result_label = 'Positif' if prediction[0] == 1 else 'Negatif'

        # Hitung akurasi
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return render_template('result.html', prediction=result_label, raw_prediction=prediction[0],k_value=jumlah_gejala, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
