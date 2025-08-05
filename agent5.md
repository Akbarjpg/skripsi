# Prompt untuk Optimasi Program Anti-Spoofing

Saya memiliki program anti-spoofing yang menggunakan 3 metode: facial landmark detection, liveness detection, dan CNN. Program sudah berjalan tetapi performanya berat dengan masalah:

1. Landmark detection mengalami delay
2. Posisi landmark kadang tidak akurat/tidak pas di wajah
3. Program terasa berat saat dijalankan

## Yang Perlu Diperbaiki:

### 1. Optimasi Landmark Detection

- Periksa apakah menggunakan model yang terlalu besar (misal: dlib 68 points vs 5 points)
- Cek resize image sebelum deteksi landmark
- Pastikan tidak ada redundant landmark detection di setiap frame
- Implementasi frame skipping jika perlu (process setiap N frame)

### 2. Optimasi CNN Model

- Periksa ukuran input image untuk CNN (resize ke ukuran optimal)
- Cek apakah model CNN terlalu kompleks untuk real-time
- Implementasi model quantization jika memungkinkan
- Gunakan model yang lebih ringan seperti MobileNet jika sesuai

### 3. Optimasi Liveness Detection

- Hindari kalkulasi yang berulang
- Cache hasil deteksi yang tidak berubah antar frame
- Gunakan threshold yang efisien

### 4. Optimasi Umum

- Implementasi multi-threading untuk proses yang independent
- Gunakan GPU acceleration jika tersedia (OpenCV DNN, CUDA)
- Hapus import library yang tidak terpakai
- Hapus fungsi/variabel yang tidak digunakan
- Optimasi loop dan conditional statements
- Gunakan numpy operations daripada python loops
- Pre-process dan cache data yang static

### 5. Memory Management

- Release memory yang tidak terpakai
- Gunakan garbage collection yang efisien
- Hindari memory leak dari OpenCV

### 6. Frame Processing Pipeline

- Process landmark, liveness, dan CNN secara pipeline, bukan sequential
- Implementasi queue system untuk frame processing
- Skip frame jika processing belum selesai

## Requirements:

- JANGAN menghilangkan salah satu dari 3 metode keamanan
- JANGAN mengurangi akurasi deteksi spoofing
- Pertahankan real-time capability (minimal 15-20 FPS)
- Code harus clean dan well-documented
- Benchmark performa sebelum dan sesudah optimasi

## Output yang Diharapkan:

1. Code yang sudah dioptimasi dengan penjelasan setiap optimasi
2. Perbandingan FPS sebelum dan sesudah
3. Memory usage comparison
4. Rekomendasi hardware
