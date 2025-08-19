# Prompt untuk Implementasi Sequential Detection Flow

Saya ingin mengubah flow detection system dari parallel (semua method jalan bersamaan) menjadi sequential (berurutan).

## CURRENT PROBLEM:

Saat ini ketiga metode (liveness, landmark, CNN) berjalan bersamaan. Ini tidak efisien dan membingungkan user.

## DESIRED FLOW:

### Phase 1: Liveness & Landmark Detection (Anti-Spoofing)

1. **Liveness Detection** - Cek apakah orang asli atau fake (foto/video)
2. **Landmark Detection** - Verifikasi gerakan natural (kedip, senyum, dll)
3. Jika KEDUA method di atas PASS → Lanjut ke Phase 2
4. Jika salah satu FAIL → Stop, tampilkan error

### Phase 2: Face Recognition (CNN)

1. Tampilkan instruksi: "Hadap lurus ke kamera untuk verifikasi identitas"
2. **CNN Face Recognition** - Cek apakah wajah terdaftar di database
3. Compare dengan face_data yang tersimpan
4. Jika match → ATTENDANCE SUCCESS
5. Jika tidak match → UNKNOWN PERSON

## IMPLEMENTATION REQUIREMENTS:

### 1. State Management

```python
class SequentialDetectionState:
    def __init__(self):
        self.phase = 'liveness'  # 'liveness' -> 'recognition' -> 'complete'
        self.liveness_passed = False
        self.landmark_passed = False
        self.anti_spoofing_passed = False
        self.recognition_result = None

    def can_proceed_to_recognition(self):
        return self.liveness_passed and self.landmark_passed
```
