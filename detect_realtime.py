"""
Script untuk deteksi ekspresi wajah secara real-time menggunakan webcam
"""

import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Import modul lokal
from preprocessing import (
    load_haarcascade, 
    detect_faces, 
    preprocess_face_for_prediction
)


class ExpressionDetector:
    """
    Class untuk mendeteksi ekspresi wajah secara real-time
    """
    
    def __init__(self, model_path='models/expression_model.h5', 
                 labels_path='class_labels.json'):
        """
        Inisialisasi detector
        
        Args:
            model_path: Path ke model yang sudah dilatih
            labels_path: Path ke file JSON berisi label kelas
        """
        print("Memuat model dan classifier...")
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
        
        self.model = load_model(model_path)
        print(f"✓ Model dimuat dari: {model_path}")
        
        # Load class labels
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.class_labels = json.load(f)
        else:
            # Default labels jika file tidak ada
            self.class_labels = {
                '0': 'marah',
                '1': 'jijik',
                '2': 'takut',
                '3': 'senang',
                '4': 'netral',
                '5': 'sedih',
                '6': 'kaget'
            }
            print(f"⚠ File {labels_path} tidak ditemukan, menggunakan label default")
        
        print(f"✓ Label kelas: {list(self.class_labels.values())}")
        
        # Load face detector (Haar Cascade)
        self.face_cascade = load_haarcascade()
        print("✓ Face detector dimuat")
        
        # Warna untuk setiap ekspresi (BGR format)
        self.colors = {
            'marah': (0, 0, 255),      # Merah
            'jijik': (0, 128, 128),    # Teal
            'takut': (128, 0, 128),    # Ungu
            'senang': (0, 255, 0),     # Hijau
            'netral': (255, 255, 255), # Putih
            'sedih': (255, 0, 0),      # Biru
            'kaget': (0, 255, 255)     # Kuning
        }
    
    def predict_expression(self, face_image):
        """
        Prediksi ekspresi dari gambar wajah
        
        Args:
            face_image: Gambar wajah (numpy array)
            
        Returns:
            expression_label: Label ekspresi yang diprediksi
            confidence: Confidence score (0-1)
        """
        # Pra-pemrosesan
        processed_face = preprocess_face_for_prediction(face_image)
        
        # Prediksi
        predictions = self.model.predict(processed_face, verbose=0)
        
        # Ambil kelas dengan probabilitas tertinggi
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Konversi index ke label
        expression_label = self.class_labels.get(str(class_idx), 'unknown')
        
        return expression_label, confidence
    
    def draw_results(self, frame, x, y, w, h, expression, confidence):
        """
        Gambar kotak dan label pada frame
        
        Args:
            frame: Frame video
            x, y, w, h: Koordinat wajah
            expression: Label ekspresi
            confidence: Confidence score
        """
        # Pilih warna berdasarkan ekspresi
        color = self.colors.get(expression, (255, 255, 255))
        
        # Tentukan ketebalan border berdasarkan confidence
        # High confidence (>70%) = thick border, Low confidence (<50%) = thin border
        thickness = 3 if confidence > 0.7 else (2 if confidence > 0.5 else 1)
        
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Siapkan teks dengan confidence score
        text = f"{expression.upper()}: {confidence*100:.1f}%"
        
        # Tambahkan warning jika confidence rendah
        if confidence < 0.6:
            text += " (?)"  # Tanda tanya untuk low confidence
        
        # Ukuran dan posisi background untuk teks
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Gambar background untuk teks
        cv2.rectangle(
            frame, 
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1
        )
        
        # Tulis teks
        cv2.putText(
            frame,
            text,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Hitam
            2
        )
    
    def run(self, camera_index=0, window_name='Deteksi Ekspresi Wajah'):
        """
        Jalankan deteksi real-time dari webcam
        
        Args:
            camera_index: Index kamera (0 untuk kamera default)
            window_name: Nama window untuk menampilkan video
        """
        print(f"\nMembuka kamera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("✗ Error: Tidak dapat membuka kamera!")
            return
        
        print("✓ Kamera terbuka")
        print("\n" + "=" * 70)
        print("DETEKSI EKSPRESI REAL-TIME")
        print("=" * 70)
        print("Tekan 'q' untuk keluar")
        print("Tekan 's' untuk screenshot")
        print("=" * 70 + "\n")
        
        frame_count = 0
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error: Tidak dapat membaca frame dari kamera")
                break
            
            frame_count += 1
            
            # Flip horizontal agar seperti cermin
            frame = cv2.flip(frame, 1)
            
            # Deteksi wajah
            faces = detect_faces(frame, self.face_cascade)
            
            # Proses setiap wajah yang terdeteksi
            for (x, y, w, h) in faces:
                # Crop wajah
                face_roi = frame[y:y+h, x:x+w]
                
                # Prediksi ekspresi
                expression, confidence = self.predict_expression(face_roi)
                
                # Gambar hasil
                self.draw_results(frame, x, y, w, h, expression, confidence)
            
            # Tampilkan jumlah wajah terdeteksi
            info_text = f"Wajah terdeteksi: {len(faces)}"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Tampilkan FPS
            fps_text = f"Frame: {frame_count}"
            cv2.putText(
                frame,
                fps_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Tampilkan frame
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nKeluar dari program...")
                break
            elif key == ord('s'):
                # Screenshot
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot disimpan: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Kamera ditutup")


def main():
    """
    Fungsi utama
    """
    # Path ke model dan labels
    model_path = 'models/expression_model.h5'
    labels_path = 'class_labels.json'
    
    # Cek apakah model ada
    if not os.path.exists(model_path):
        print(f"Error: Model tidak ditemukan di {model_path}")
        print("Silakan latih model terlebih dahulu dengan menjalankan: python src/train.py")
        sys.exit(1)
    
    try:
        # Buat detector
        detector = ExpressionDetector(model_path, labels_path)
        
        # Jalankan deteksi real-time
        detector.run(camera_index=0)
        
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
