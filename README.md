# Deteksi Ekspresi Wajah Real-time dengan CNN

Sistem deteksi ekspresi wajah menggunakan Convolutional Neural Network (CNN) yang dibangun dari awal tanpa pre-trained model. Model dapat mendeteksi 7 ekspresi wajah: marah, jijik, takut, senang, netral, sedih, dan kaget.

## Fitur

- CNN Custom - Dibangun dari nol tanpa transfer learning
- Data Augmentation - Rotasi, zoom, shift untuk meningkatkan performa
- Real-time Detection - Deteksi ekspresi dari webcam
- Multi-face Support - Dapat mendeteksi beberapa wajah sekaligus
- Visualisasi Training - Grafik akurasi dan loss otomatis

## Struktur Proyek

```
ekspresi-wajah/
 dataset/
    train/              (Data training 80%)
       marah/
       jijik/
       takut/
       senang/
       netral/
       sedih/
       kaget/
    validation/         (Data validasi 20%)
        marah/
        jijik/
        takut/
        senang/
        netral/
        sedih/
        kaget/
 src/
    model.py            (Arsitektur CNN)
    preprocessing.py    (Preprocessing & augmentasi)
    train.py            (Script training)
    detect_realtime.py  (Real-time detection)
    predict_image.py    (Prediksi gambar)
    evaluate.py         (Evaluasi model)
    test_setup.py       (Verifikasi instalasi)
 models/                 (Model terlatih)
 requirements.txt
 README.md
```

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Razin907/ekspresi-wajah.git
cd ekspresi-wajah
```

### 2. Buat Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Verifikasi Instalasi

```powershell
python src/test_setup.py
```

## Persiapan Dataset

### Download Dataset FER2013

1. Download dari Kaggle FER2013: https://www.kaggle.com/datasets/msambare/fer2013
2. Ekstrak file zip
3. Pindahkan gambar dengan mapping:

| Dataset FER2013 | Folder Proyek |
|-----------------|---------------|
| angry           | marah/        |
| disgust         | jijik/        |
| fear            | takut/        |
| happy           | senang/       |
| neutral         | netral/       |
| sad             | sedih/        |
| surprise        | kaget/        |

Tips: Gunakan 80% untuk training, 20% untuk validation. Minimal 500-1000 gambar per kelas untuk hasil optimal.

## Training Model

### Mulai Training

```powershell
python src/train.py
```

### Output Training

- models/expression_model.h5  (Model terlatih)
- class_labels.json            (Mapping label)
- training_history.png         (Grafik akurasi & loss)
- training_log.csv             (Log detail)

### Parameter Default

- epochs: 50
- batch_size: 64
- learning_rate: 0.001
- input_shape: (48, 48, 1) grayscale

## Testing Model

### Real-time Detection

```powershell
python src/detect_realtime.py
```

Kontrol: q untuk keluar, s untuk screenshot

### Prediksi Gambar

```powershell
python src/predict_image.py path/to/image.jpg
```

### Evaluasi Model

```powershell
python src/evaluate.py
```

## Arsitektur Model CNN

```
INPUT: 48x48x1 Grayscale

CONV BLOCK 1
- Conv2D (32 filters) + ReLU
- BatchNormalization
- Conv2D (32 filters) + ReLU
- BatchNormalization
- MaxPooling2D (2x2)
- Dropout (0.25)

CONV BLOCK 2
- Conv2D (64 filters) + ReLU
- BatchNormalization
- Conv2D (64 filters) + ReLU
- BatchNormalization
- MaxPooling2D (2x2)
- Dropout (0.25)

CONV BLOCK 3
- Conv2D (128 filters) + ReLU
- BatchNormalization
- Conv2D (128 filters) + ReLU
- BatchNormalization
- MaxPooling2D (2x2)
- Dropout (0.25)

CONV BLOCK 4
- Conv2D (256 filters) + ReLU
- BatchNormalization
- MaxPooling2D (2x2)
- Dropout (0.25)

CLASSIFIER
- Flatten
- Dense (256) + ReLU + Dropout (0.5)
- Dense (128) + ReLU + Dropout (0.5)
- Dense (7) + Softmax

OUTPUT: 7 classes
```

Total Parameters: ~2-3 juta

## Konfigurasi

### Data Augmentation

- Rotation: 15
- Width/Height Shift: 15%
- Shear: 15%
- Zoom: 15%
- Horizontal Flip: Yes

### Training Callbacks

- ModelCheckpoint: Simpan model terbaik
- EarlyStopping: Stop jika tidak improvement (patience=10)
- ReduceLROnPlateau: Kurangi learning rate (patience=5)
- CSVLogger: Log training history

## Tips Meningkatkan Akurasi

1. Tambah Data - Semakin banyak data, semakin baik (target 1000+ per kelas)
2. Balance Dataset - Pastikan jumlah seimbang per kelas
3. Augmentasi Lebih Agresif - Eksperimen parameter di preprocessing.py
4. Hyperparameter Tuning - Coba berbagai learning rate dan batch size
5. Training Lebih Lama - Tambah epochs jika belum converge
6. Arsitektur Lebih Kompleks - Tambah layer atau filter di model.py

## Troubleshooting

### Model tidak ditemukan

Jalankan training terlebih dahulu:
```powershell
python src/train.py
```

### Dataset tidak ditemukan

Periksa struktur folder dataset sesuai dokumentasi di atas.

### Akurasi rendah

Kemungkinan: Dataset terlalu kecil, tidak seimbang, atau training terlalu sebentar.
Solusi: Tambah data, balance dataset, tambah epochs, sesuaikan learning rate.

### Kamera tidak terbuka

Pastikan webcam terhubung, tutup aplikasi lain yang menggunakan webcam.
Coba ubah camera_index=0 menjadi camera_index=1 di detect_realtime.py

### Out of Memory

Kurangi batch_size (dari 64 ke 32 atau 16), kurangi ukuran gambar, tutup aplikasi lain.

## Catatan Penting

Haar Cascade untuk deteksi wajah BUKAN bagian dari model CNN!
- Haar Cascade hanya untuk preprocessing (menemukan dan crop wajah)
- Model CNN fokus pada klasifikasi ekspresi
- Semua layer CNN dibangun dan dilatih dari awal

Alur kerja:
Gambar  Haar Cascade (deteksi wajah)  Crop  Resize  Normalisasi  Model CNN (klasifikasi)  Output

## Use Cases

- Pendidikan: Pembelajaran Deep Learning & Computer Vision
- Research: Analisis ekspresi wajah dan emosi
- Business: Customer sentiment analysis
- Healthcare: Patient emotion monitoring
- Gaming: Emotion-based interaction
- Security: Emotion-based authentication

## Referensi

- Keras Documentation: https://keras.io/
- TensorFlow: https://www.tensorflow.org/
- OpenCV: https://docs.opencv.org/
- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013

## Author

Razin907
- GitHub: @Razin907
- Repository: https://github.com/Razin907/ekspresi-wajah

## Contributing

Kontribusi sangat diterima! Silakan baca CONTRIBUTING.md untuk panduan lengkap.

Area kontribusi: Bug fixes, Fitur baru, Dokumentasi, Testing, UI/UX improvements

## License

Proyek ini dilisensikan di bawah MIT License.
Lihat file LICENSE untuk detail lengkap.
Bebas digunakan untuk pembelajaran, penelitian, atau komersial.

## Support

Jika proyek ini bermanfaat, berikan star di GitHub!

Untuk pertanyaan atau masalah:
- Buat Issue: https://github.com/Razin907/ekspresi-wajah/issues
- Discussion: https://github.com/Razin907/ekspresi-wajah/discussions

---

Selamat Mencoba!
Made with Love for Computer Vision & Deep Learning
