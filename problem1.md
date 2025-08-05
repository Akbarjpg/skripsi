# Prompt untuk Memperbaiki Security Assessment dan Detection Methods

Program anti-spoofing saya memiliki masalah dengan sistem security assessment yang tidak stabil:

## MASALAH UTAMA:

### 1. Security Assessment Reset Terus-Menerus

- Status selalu reset ke default (merah/false)
- Padahal seharusnya mempertahankan status hijau setelah verifikasi berhasil
- Minimal 2 dari 3 method harus hijau untuk lolos

### 2. Movement Detection Terlalu Sensitif

- Hijau saat bergerak, langsung merah saat berhenti sebentar
- Tidak realistis - user tidak mungkin gerak terus-menerus
- Seharusnya ada grace period atau memory state

### 3. Challenge/Instruction Tidak Muncul

- Tidak ada instruksi untuk user (hadap kanan/kiri, kedip, dll)
- User bingung harus ngapain untuk verifikasi
- Challenge system sepertinya tidak aktif atau tidak terintegrasi

### 4. Method Lain Tidak Aktif

- Hanya movement detection yang responsif
- CNN dan landmark detection sepertinya tidak update status
- Atau mungkin threshold terlalu tinggi

## ANALISIS ROOT CAUSE:

1. **State Management Issue**

   - Status detection methods tidak persistent
   - Setiap frame di-reset instead of maintaining state
   - Tidak ada time window untuk evaluasi

2. **Threshold Configuration**

   - Movement threshold mungkin terlalu strict
   - CNN confidence threshold mungkin terlalu tinggi
   - Landmark detection criteria unclear

3. **Challenge System Not Integrated**
   - Random challenge generator tidak dipanggil
   - UI tidak menampilkan instruksi
   - Verification logic tidak memeriksa challenge completion

## SOLUSI YANG DIPERLUKAN:

### 1. Implement State Persistence

```python
class SecurityAssessmentState:
    def __init__(self):
        self.movement_verified = False
        self.movement_last_verified = None
        self.movement_grace_period = 3.0  # seconds

        self.cnn_verified = False
        self.cnn_confidence_history = []

        self.landmark_verified = False
        self.landmark_challenge_completed = False

    def update_movement(self, is_moving):
        if is_moving:
            self.movement_verified = True
            self.movement_last_verified = time.time()
        elif self.movement_verified:
            # Keep verified status for grace period
            if time.time() - self.movement_last_verified < self.movement_grace_period:
                return True
        return self.movement_verified
```

### 2. Fix Challenge System Integration

- Generate random challenges (blink, turn head, smile)
- Display clear instructions to user
- Track challenge completion
- Update landmark_verified based on challenge

### 3. Adjust Detection Thresholds

- Movement: Add grace period (3 seconds after movement stops)
- CNN: Lower confidence threshold to 0.7 (from maybe 0.9)
- Landmark: Clear pass criteria based on challenge completion

### 4. Improve Status Display

- Show which methods are verified with checkmarks
- Display current challenge instruction prominently
- Show progress/countdown for challenges
- Maintain green status once verified (don't reset)

## EXPECTED BEHAVIOR:

1. **Movement Detection:**

   - Goes green when user moves
   - STAYS green for 3 seconds after stopping
   - Only turns red if no movement for > 3 seconds

2. **CNN Detection:**

   - Evaluates average confidence over 1 second
   - Goes green if average > 0.7
   - Stays green once verified

3. **Landmark Detection:**

   - Shows challenge instruction (e.g., "Please blink twice")
   - Goes green when challenge completed
   - Rotates through different challenges

4. **Overall Security:**
   - Pass if 2/3 methods are green
   - Status persists (no constant reset)
   - Clear feedback to user

## IMPLEMENTATION REQUIREMENTS:

1. **Update Frontend (JavaScript):**

   - Display challenge instructions
   - Show persistent status (not resetting)
   - Add visual progress indicators

2. **Update Backend (Python):**

   - Implement state management class
   - Add challenge generation and verification
   - Fix threshold values
   - Add grace periods and time windows

3. **Integration:**
   - Ensure all changes work with existing codebase
   - Test with real webcam input
   - Verify all 3 methods can achieve green status

## TESTING CHECKLIST:

- [ ] Movement detection stays green for 3 seconds after stopping
- [ ] CNN detection achieves green with normal face
- [ ] Challenge instructions appear and are verifiable
- [ ] 2/3 methods green = overall pass
- [ ] Status doesn't constantly reset
- [ ] User experience is smooth and clear

Tolong implementasikan perbaikan ini langsung ke main program sehingga security assessment berfungsi dengan baik dan user-
