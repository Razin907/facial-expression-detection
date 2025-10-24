"""
Script untuk pra-pemrosesan data dan augmentasi
"""

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(train_dir, validation_dir, target_size=(48, 48), 
                           batch_size=64, augmentation=True):
    """
    Membuat data generator untuk training dan validation
    
    Args:
        train_dir: Path ke folder training
        validation_dir: Path ke folder validation
        target_size: Ukuran target gambar (height, width)
        batch_size: Ukuran batch untuk training
        augmentation: Apakah menggunakan augmentasi atau tidak
        
    Returns:
        train_generator, validation_generator
    """
    
    if augmentation:
        # Data Generator untuk Training dengan Augmentasi
        train_datagen = ImageDataGenerator(
            rescale=1./255,                    # Normalisasi pixel [0, 255] -> [0, 1]
            rotation_range=15,                 # Rotasi acak hingga 15 derajat
            width_shift_range=0.15,            # Geser horizontal 15%
            height_shift_range=0.15,           # Geser vertikal 15%
            shear_range=0.15,                  # Transformasi shear
            zoom_range=0.15,                   # Zoom in/out
            horizontal_flip=True,              # Balik horizontal
            fill_mode='nearest'                # Mengisi pixel yang kosong
        )
    else:
        # Tanpa augmentasi, hanya normalisasi
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data Generator untuk Validation (hanya normalisasi, tanpa augmentasi)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',              # Grayscale untuk ekspresi wajah
        class_mode='categorical',            # Multi-class classification
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False                        # Tidak perlu shuffle untuk validation
    )
    
    return train_generator, validation_generator


def preprocess_face_for_prediction(face_image, target_size=(48, 48)):
    """
    Pra-pemrosesan gambar wajah untuk prediksi
    
    Args:
        face_image: Gambar wajah (numpy array)
        target_size: Ukuran target (height, width)
        
    Returns:
        Gambar yang sudah diproses dan siap untuk prediksi
    """
    # Ubah ke grayscale jika belum
    if len(face_image.shape) == 3:
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_image
    
    # Resize ke ukuran target
    face_resized = cv2.resize(face_gray, target_size)
    
    # Normalisasi [0, 255] -> [0, 1]
    face_normalized = face_resized / 255.0
    
    # Reshape untuk input model: (1, height, width, 1)
    face_processed = face_normalized.reshape(1, target_size[0], target_size[1], 1)
    
    return face_processed


def load_haarcascade():
    """
    Memuat Haar Cascade classifier untuk deteksi wajah
    
    Returns:
        Face cascade classifier
    """
    # Path ke Haar Cascade (OpenCV menyediakan file ini)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise IOError("Tidak dapat memuat Haar Cascade untuk deteksi wajah")
    
    return face_cascade


def detect_faces(image, face_cascade, scale_factor=1.1, min_neighbors=5):
    """
    Mendeteksi wajah dalam gambar menggunakan Haar Cascade
    
    Args:
        image: Gambar input
        face_cascade: Haar Cascade classifier
        scale_factor: Parameter untuk deteksi multi-scale
        min_neighbors: Minimum tetangga untuk deteksi
        
    Returns:
        List koordinat wajah [(x, y, w, h), ...]
    """
    # Convert ke grayscale jika belum
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    return faces


def print_dataset_info(train_generator, validation_generator):
    """
    Menampilkan informasi dataset
    """
    print("\n" + "=" * 70)
    print("INFORMASI DATASET")
    print("=" * 70)
    print(f"Jumlah data training: {train_generator.n}")
    print(f"Jumlah data validation: {validation_generator.n}")
    print(f"Jumlah kelas: {train_generator.num_classes}")
    print(f"Nama kelas: {list(train_generator.class_indices.keys())}")
    print(f"Ukuran batch: {train_generator.batch_size}")
    print(f"Ukuran gambar: {train_generator.target_size}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    # Test preprocessing
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Membuat data generators...")
        train_gen, val_gen = create_data_generators(train_dir, val_dir)
        print_dataset_info(train_gen, val_gen)
    else:
        print("Dataset belum tersedia. Silakan tambahkan data ke folder dataset/")
    
    # Test Haar Cascade
    print("Testing Haar Cascade untuk deteksi wajah...")
    try:
        face_cascade = load_haarcascade()
        print("✓ Haar Cascade berhasil dimuat!")
    except Exception as e:
        print(f"✗ Error: {e}")
