# Agent Prompt: Sistem Absensi Berbasis Pengenalan Wajah dengan Anti-Spoofing

## Deskripsi Proyek

Saya sedang mengembangkan sistem absensi berbasis pengenalan wajah yang menggunakan pendekatan hybrid dengan tiga komponen utama:

1. CNN (Convolutional Neural Network) untuk liveness detection
2. Facial Landmark Detection untuk verifikasi gerakan alami
3. Challenge-Response mechanism untuk mencegah spoofing

## Tujuan Sistem

- Membangun aplikasi web absensi yang aman dan akurat
- Mendeteksi dan mencegah kecurangan menggunakan foto, video, atau livestreaming (Zoom/Google Meet)
- Memverifikasi kehadiran fisik pengguna secara real-time
- Implementasi multi-layer security dengan logika fusion (minimal 2 dari 3 metode harus lulus)

## Dataset yang Tersedia

Saya memiliki dataset di folder `test_img` dengan struktur:

- `/color/` - berisi gambar RGB dengan format: `{video_id}_{frame_number}_{label}.jpg`
  - Label: "real" untuk wajah asli, "fake" untuk wajah palsu
- `/depth/` - berisi gambar depth (jika ada)

## Komponen yang Perlu Dibangun

### 1. CNN Model untuk Liveness Detection

- Arsitektur CNN yang efisien untuk klasifikasi real/fake faces
- Data preprocessing dan augmentation untuk dataset yang ada
- Training pipeline dengan monitoring metrics
- Model evaluation dan optimization

### 2. Facial Landmark Detection

- Implementasi deteksi 68 landmark points menggunakan dlib/MediaPipe
- Algoritma untuk verifikasi gerakan alami:
  - Eye blink detection
  - Mouth opening detection
  - Head movement tracking (yaw, pitch, roll)
- Threshold adjustment untuk berbagai kondisi pencahayaan

### 3. Challenge-Response System

- Random challenge generator dengan variasi:
  - "Kedip 2 kali"
  - "Buka mulut"
  - "Putar kepala ke kanan/kiri"
  - "Angguk"
  - Kombinasi gerakan
- Real-time response verification
- Anti-replay attack mechanism

### 4. Web Application

- Frontend dengan webcam integration
- Real-time face detection dan tracking
- User interface untuk challenge display
- Backend API untuk:
  - Face enrollment
  - Attendance recording
  - Verification processing
  - Database management

### 5. Additional Features

- Geolocation verification
- IP address logging
- Session management
- Attendance reports dan analytics
- Admin dashboard

## Tech Stack yang Direkomendasikan

- **Backend**: Python (Flask/FastAPI)
- **ML Framework**: TensorFlow/PyTorch
- **Face Detection**: OpenCV, dlib, MediaPipe
- **Frontend**: React/Vue.js dengan WebRTC
- **Database**: PostgreSQL/MongoDB
- **Deployment**: Docker, cloud services

## Pertanyaan Spesifik untuk Development

1. **CNN Architecture**: Bagaimana cara mendesain arsitektur CNN yang optimal untuk dataset saya? Berapa layer yang dibutuhkan dan hyperparameter apa yang harus di-tune?

2. **Data Preprocessing**: Bagaimana cara melakukan preprocessing pada gambar di folder `test_img`? Apakah perlu normalisasi, resizing, atau augmentasi khusus?

3. **Training Strategy**: Dengan dataset yang ada, bagaimana strategi training yang baik? Berapa split ratio untuk train/validation/test? Bagaimana mengatasi potential overfitting?

4. **Landmark Detection Integration**: Bagaimana cara mengintegrasikan facial landmark detection dengan CNN model? Apakah sebaiknya diproses secara paralel atau sequential?

5. **Real-time Performance**: Bagaimana mengoptimalkan sistem agar berjalan real-time di browser? Pertimbangan apa saja untuk latency dan computational load?

6. **Anti-Spoofing Robustness**: Bagaimana memastikan sistem dapat mendeteksi spoofing dari high-quality displays atau video calls? Fitur apa yang paling diskriminatif?

7. **Challenge Validation**: Bagaimana implementasi algoritma untuk memvalidasi response dari challenge? Berapa tolerance threshold yang optimal?

8. **Security Considerations**: Apa saja potential security vulnerabilities dan bagaimana mengatasinya? Bagaimana mencegah model dari adversarial attacks?

9. **Scalability**: Bagaimana mendesain sistem agar scalable untuk banyak user? Pertimbangan untuk concurrent requests dan database optimization?

10. **Testing Strategy**: Bagaimana strategi testing yang komprehensif? Unit tests, integration tests, dan real-world scenario tests apa yang diperlukan?

## Output yang Diharapkan

1. Complete codebase dengan modular architecture
2. Trained CNN model dengan high accuracy
3. Robust facial landmark detection system
4. Secure challenge-response mechanism
5. User-friendly web application
6. Comprehensive documentation
7. Deployment guide

Mohon bantu saya mengembangkan sistem ini step by step, dimulai dari pembuatan CNN model menggunakan dataset yang ada di folder `test_img`.
