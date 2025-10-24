# Deteksi Ekspresi Wajah Real-time dengan CNN

🎭 **Ready-to-Use!** Sistem deteksi ekspresi wajah menggunakan Convolutional Neural Network (CNN) yang sudah dilatih dan siap pakai. Model dapat mendeteksi 7 ekspresi wajah: **marah**, **jijik**, **takut**, **senang**, **netral**, **sedih**, dan **kaget**.

## ✨ Fitur

- ✅ **Langsung Pakai** - Model sudah dilatih, tidak perlu training lagi!
- 🎥 **Real-time Detection** - Deteksi ekspresi dari webcam
- 👥 **Multi-face Support** - Dapat mendeteksi beberapa wajah sekaligus
- 🎨 **Color-coded Labels** - Setiap ekspresi punya warna berbeda
- 📸 **Screenshot Support** - Simpan hasil deteksi dengan tekan 's'
- 🖼️ **Image Prediction** - Prediksi ekspresi dari gambar

## 📁 Struktur Proyek

```
ekspresi-wajah-demo/
├── models/
│   ├── expression_model.h5              # ✅ Model CNN terlatih (14 MB)
│   ├── class_labels.json                # ✅ Mapping 7 kelas ekspresi
│   └── haarcascade_frontalface_default.xml  # ✅ Face detector
├── detect_realtime.py                   # 🎥 Script deteksi real-time
├── predict_image.py                     # 🖼️ Script prediksi gambar
├── preprocessing.py                     # 🔧 Helper functions
├── requirements.txt                     # 📦 Dependencies
├── README.md                            # 📖 Dokumentasi
└── LICENSE                              # ⚖️ MIT License
```

## 🚀 Quick Start (3 Langkah!)

### 1. Clone Repository

```powershell
git clone https://github.com/Razin907/facial-expression-detection.git
cd facial-expression-detection
```

### 2. Install Dependencies

```powershell
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Jalankan Deteksi!

**Deteksi Real-time dari Webcam:**
```powershell
python detect_realtime.py
```

**Atau Prediksi dari Gambar:**
```powershell
python predict_image.py path\to\your\image.jpg
```

## 🎮 Cara Penggunaan

### Deteksi Real-time (Webcam)

```powershell
python detect_realtime.py
```

**Kontrol:**
- Tekan **`q`** untuk keluar
- Tekan **`s`** untuk screenshot (disimpan sebagai `screenshot_*.jpg`)

**Output:**
- Kotak berwarna di sekitar wajah
- Label ekspresi + confidence score
- Jumlah wajah terdeteksi
- Frame counter

### Prediksi dari Gambar

```powershell
python predict_image.py path\to\image.jpg
```

**Output:**
- Gambar hasil dengan anotasi disimpan sebagai `*_result.jpg`
- Tampilan preview hasil prediksi
- Informasi ekspresi + confidence di terminal

**Contoh:**
```powershell
python predict_image.py foto_saya.jpg
# Output: foto_saya_result.jpg
```

## 🎨 Label & Warna Ekspresi

| Ekspresi | Label | Warna |
|----------|-------|-------|
| 😠 Marah | `marah` | 🔴 Merah |
| 🤢 Jijik | `jijik` | 🟦 Teal |
| 😨 Takut | `takut` | 🟣 Ungu |
| 😊 Senang | `senang` | 🟢 Hijau |
| 😐 Netral | `netral` | ⚪ Putih |
| 😢 Sedih | `sedih` | 🔵 Biru |
| 😲 Kaget | `kaget` | 🟡 Kuning |

## 🧠 Tentang Model

**Arsitektur:** Custom CNN (Convolutional Neural Network)
**Input:** 48x48 grayscale images
**Output:** 7 kelas ekspresi
**Parameters:** ~2-3 juta
**Training:** Dilatih dari awal tanpa transfer learning

### Arsitektur CNN

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

OUTPUT: 7 classes (marah, jijik, takut, senang, netral, sedih, kaget)
```

**Alur Kerja:**
```
Input Image → Haar Cascade (deteksi wajah) → Crop & Resize (48x48) 
→ Grayscale → Normalisasi → CNN Model → Prediksi Ekspresi
```

## 📦 Dependencies

- **TensorFlow** 2.20.0 - Deep learning framework
- **OpenCV** ≥4.8.0 - Computer vision & face detection
- **NumPy** ≥1.24.0 - Array operations
- **Matplotlib** ≥3.8.0 - Visualization (opsional)
- **scikit-learn** ≥1.3.0 - Metrics (opsional)

Install semua dengan:
```powershell
pip install -r requirements.txt
```

## 🔧 Troubleshooting

### Kamera tidak terbuka
**Masalah:** Webcam tidak terdeteksi atau error saat membuka kamera

**Solusi:**
- Pastikan webcam terhubung dan tidak digunakan aplikasi lain
- Tutup aplikasi lain yang menggunakan webcam (Zoom, Teams, dll)
- Coba ubah `camera_index=0` menjadi `camera_index=1` di `detect_realtime.py` line 265

### Error "Model tidak ditemukan"
**Masalah:** `FileNotFoundError: Model tidak ditemukan di: models/expression_model.h5`

**Solusi:**
- Pastikan file `models/expression_model.h5` ada (ukuran ~14 MB)
- Pastikan Anda menjalankan script dari root directory project
- Clone ulang repository jika model tidak ada

### Gambar tidak terdeteksi wajahnya
**Masalah:** Tidak ada kotak deteksi muncul di webcam/gambar

**Solusi:**
- Pastikan wajah cukup terang dan menghadap kamera
- Jarak wajah tidak terlalu dekat atau terlalu jauh
- Coba adjust parameter `scale_factor` dan `min_neighbors` di `preprocessing.py`

### Import Error
**Masalah:** `ModuleNotFoundError: No module named 'tensorflow'` atau library lain

**Solusi:**
```powershell
pip install -r requirements.txt
```

### Out of Memory (OOM)
**Masalah:** Error memory saat run di komputer low-spec

**Solusi:**
- Tutup aplikasi lain yang berat
- Kurangi resolusi webcam jika perlu
- Model sudah cukup ringan (~14 MB) untuk CPU biasa

## 📝 Catatan Penting

- **Haar Cascade** untuk deteksi wajah BUKAN bagian dari model CNN!
  - Haar Cascade = preprocessing (menemukan & crop wajah)
  - CNN Model = klasifikasi ekspresi wajah
- **Model sudah dilatih** - Tidak perlu download dataset atau training ulang
- **Label dalam Bahasa Indonesia** - Output menggunakan label Indonesia
- **Webcam mirror mode** - Frame di-flip horizontal untuk efek cermin

## 💡 Use Cases

- 🎓 **Pendidikan** - Pembelajaran Deep Learning & Computer Vision
- 🔬 **Research** - Analisis ekspresi wajah dan emosi
- 💼 **Business** - Customer sentiment analysis real-time
- 🏥 **Healthcare** - Patient emotion monitoring
- 🎮 **Gaming** - Emotion-based game interaction
- 🔐 **Security** - Emotion-based behavior analysis

## 📚 Referensi

- [Keras Documentation](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) - Dataset yang digunakan untuk training

## 👨‍💻 Author

**Razin907**
- GitHub: [@Razin907](https://github.com/Razin907)
- Repository: [facial-expression-detection](https://github.com/Razin907/facial-expression-detection)

## 🤝 Contributing

Kontribusi sangat diterima! Ada banyak cara untuk berkontribusi:

- 🐛 **Bug Reports** - Laporkan bug yang Anda temukan
- ✨ **Feature Requests** - Sarankan fitur baru
- 📖 **Documentation** - Perbaiki atau tambah dokumentasi
- 🧪 **Testing** - Bantu test di berbagai environment
- 🎨 **UI/UX** - Perbaiki tampilan atau user experience

Silakan buka [Issues](https://github.com/Razin907/facial-expression-detection/issues) atau [Pull Request](https://github.com/Razin907/facial-expression-detection/pulls)!

## ⚖️ License

Proyek ini dilisensikan di bawah **MIT License**.

Lihat file [LICENSE](LICENSE) untuk detail lengkap.

**Bebas digunakan untuk:**
- ✅ Pembelajaran & penelitian
- ✅ Proyek pribadi
- ✅ Proyek komersial
- ✅ Modifikasi & distribusi

## ⭐ Support

Jika proyek ini bermanfaat, berikan ⭐ **star** di GitHub!

**Ada pertanyaan atau masalah?**
- 💬 [GitHub Issues](https://github.com/Razin907/facial-expression-detection/issues)
- 🗨️ [GitHub Discussions](https://github.com/Razin907/facial-expression-detection/discussions)

---

<div align="center">

**🎭 Selamat Mencoba! 🎭**

Made with ❤️ for Computer Vision & Deep Learning

*Happy Coding!* 🚀

</div>
