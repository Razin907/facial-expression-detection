"""
Script untuk melakukan prediksi pada gambar tunggal
"""

import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import (
    load_haarcascade, 
    detect_faces, 
    preprocess_face_for_prediction
)


def predict_image(image_path, 
                  model_path='models/expression_model.h5',
                  labels_path='class_labels.json',
                  save_result=True):
    """
    Prediksi ekspresi pada gambar tunggal
    
    Args:
        image_path: Path ke gambar
        model_path: Path ke model
        labels_path: Path ke class labels
        save_result: Apakah menyimpan hasil
    """
    # Load model
    print(f"Memuat model dari: {model_path}")
    model = load_model(model_path)
    
    # Load labels
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            class_labels = json.load(f)
    else:
        class_labels = {
            '0': 'marah',
            '1': 'jijik',
            '2': 'takut',
            '3': 'senang',
            '4': 'netral',
            '5': 'sedih',
            '6': 'kaget'
        }
    
    # Load face detector
    face_cascade = load_haarcascade()
    
    # Load gambar
    print(f"Memuat gambar dari: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Tidak dapat memuat gambar dari {image_path}")
        return
    
    # Buat copy untuk menggambar hasil
    result_image = image.copy()
    
    # Deteksi wajah
    faces = detect_faces(image, face_cascade)
    print(f"Wajah terdeteksi: {len(faces)}")
    
    if len(faces) == 0:
        print("Tidak ada wajah yang terdeteksi!")
        return
    
    # Warna untuk setiap ekspresi
    colors = {
        'marah': (0, 0, 255),
        'jijik': (0, 128, 128),
        'takut': (128, 0, 128),
        'senang': (0, 255, 0),
        'netral': (255, 255, 255),
        'sedih': (255, 0, 0),
        'kaget': (0, 255, 255)
    }
    
    # Proses setiap wajah
    for i, (x, y, w, h) in enumerate(faces):
        # Crop wajah
        face_roi = image[y:y+h, x:x+w]
        
        # Pra-pemrosesan
        processed_face = preprocess_face_for_prediction(face_roi)
        
        # Prediksi
        predictions = model.predict(processed_face, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Konversi ke label
        expression = class_labels.get(str(class_idx), 'unknown')
        
        print(f"\nWajah {i+1}:")
        print(f"  Ekspresi: {expression.upper()}")
        print(f"  Confidence: {confidence*100:.2f}%")
        
        # Warning jika confidence rendah
        if confidence < 0.6:
            print(f"  ⚠️  Low confidence - hasil mungkin tidak akurat")
            print(f"  💡 Tips: Gunakan lighting lebih baik atau ekspresi lebih jelas")
        
        # Gambar hasil
        color = colors.get(expression, (255, 255, 255))
        
        # Tentukan ketebalan border berdasarkan confidence
        thickness = 3 if confidence > 0.7 else (2 if confidence > 0.5 else 1)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, thickness)
        
        # Tambahkan tanda tanya jika confidence rendah
        text = f"{expression.upper()}: {confidence*100:.1f}%"
        if confidence < 0.6:
            text += " (?)"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(
            result_image,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1
        )
        
        cv2.putText(
            result_image,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    # Tampilkan hasil
    cv2.imshow('Hasil Prediksi', result_image)
    print("\nTekan sembarang tombol untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Simpan hasil
    if save_result:
        output_path = image_path.replace('.', '_result.')
        cv2.imwrite(output_path, result_image)
        print(f"✓ Hasil disimpan di: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py <path_to_image>")
        print("Example: python src/predict_image.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} tidak ditemukan!")
        sys.exit(1)
    
    model_path = 'models/expression_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        print("Silakan latih model terlebih dahulu dengan: python src/train.py")
        sys.exit(1)
    
    predict_image(image_path)
