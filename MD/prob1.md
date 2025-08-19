# Prompt untuk Implementasi Sistem Registrasi Wajah di Dashboard

Saya membutuhkan sistem registrasi wajah untuk pendaftaran absensi. Saat ini sistem absensi wajah tidak bisa digunakan karena belum ada data wajah user yang terdaftar.

## REQUIREMENTS:

### 1. Tambahkan Menu di Dashboard

- Tambahkan tombol/card "Daftarkan Wajah" di dashboard
- Hanya untuk user yang belum punya data wajah
- Cek apakah user sudah punya data wajah atau belum

### 2. Halaman Registrasi Wajah (/register-face)

- Capture 3 posisi wajah:
  - **Depan** (frontal face)
  - **Kanan** (right profile)
  - **Kiri** (left profile)
- Guide/instruksi untuk setiap posisi
- Progress indicator (1/3, 2/3, 3/3)
- Preview captured images sebelum save

### 3. Database Schema untuk Face Data

```sql
CREATE TABLE IF NOT EXISTS face_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    face_position TEXT NOT NULL, -- 'front', 'right', 'left'
    face_encoding TEXT NOT NULL, -- Face embeddings/features
    image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### 4. Face Registration Flow

```
Dashboard → Click "Daftarkan Wajah" → Registration Page
↓
Step 1: Capture Front Face
- Show guide overlay
- Countdown 3-2-1
- Capture & extract features
- Show preview
↓
Step 2: Capture Right Face
- Instruksi "Hadap ke kanan"
- Same capture process
↓
Step 3: Capture Left Face
- Instruksi "Hadap ke kiri"
- Same capture process
↓
Review & Save
- Show all 3 captured images
- Option to retake
- Save to database
↓
Success → Redirect to Dashboard
```

### 5. Backend Implementation

#### Route: Check Face Registration Status

```python
@app.route('/api/check-face-registered')
def check_face_registered():
    if 'user_id' not in session:
        return jsonify({'registered': False})

    # Check if user has face data
    user_id = session['user_id']
    # Query database for face_data
    # Return status
```

#### Route: Register Face Page

```python
@app.route('/register-face')
def register_face():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Check if already registered
    # If yes, redirect to dashboard
    # If no, show registration page
```

#### SocketIO: Handle Face Capture

```python
@socketio.on('capture_face')
def handle_capture_face(data):
    position = data['position']  # 'front', 'right', 'left'
    image_data = data['image']
    user_id = session.get('user_id')

    # Extract face features using CNN
    # Save to database
    # Return success/failure
```

### 6. Frontend Implementation

#### Dashboard Update

```html
<!-- Add to dashboard.html -->
{% if not user_has_face_data %}
<div class="col-md-6 col-lg-4 mb-4">
  <div class="card bg-warning text-white">
    <div class="card-body">
      <h5 class="card-title">
        <i class="fas fa-user-plus"></i> Daftarkan Wajah
      </h5>
      <p class="card-text">
        Anda belum mendaftarkan wajah. Daftarkan sekarang untuk bisa absen.
      </p>
      <a href="{{ url_for('register_face') }}" class="btn btn-light">
        Daftar Sekarang
      </a>
    </div>
  </div>
</div>
{% endif %}
```

#### Face Registration Page

```html
<!-- register_face.html -->
<div class="registration-wizard">
  <div class="progress mb-4">
    <div class="progress-bar" id="progressBar" style="width: 33%"></div>
  </div>

  <div class="step" id="step1">
    <h3>Langkah 1: Wajah Depan</h3>
    <div class="camera-container">
      <video id="video" autoplay></video>
      <canvas id="canvas" style="display:none;"></canvas>
      <div class="face-guide-overlay">
        <!-- SVG overlay untuk guide posisi wajah -->
      </div>
    </div>
    <button onclick="captureface('front')" class="btn btn-primary">
      Ambil Foto (3)
    </button>
  </div>

  <!-- Similar for step2 (right) and step3 (left) -->
</div>
```

### 7. Face Feature Extraction

```python
def extract_face_features(image):
    """Extract face embeddings using CNN model"""
    # Use face_recognition or similar library
    # Return 128-dimensional face encoding
    # Store as JSON string in database
```

### 8. Integration with Attendance System

- Modify attendance face detection to compare with stored face data
- Use face matching algorithm (cosine similarity, euclidean distance)
- Set appropriate threshold for matching

### 9. Admin Account Fix

```python
# Fix admin login issue in login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Special handling for admin
        if username == 'admin' and password == 'admin':
            # Create admin session directly
            session['user_id'] = 0  # Special admin ID
            session['username'] = 'admin'
            session['full_name'] = 'Administrator'
            session['role'] = 'admin'
            return redirect(url_for('dashboard'))
```

## EXPECTED OUTCOME:

1. **Dashboard Enhancement:**

   - Shows "Daftarkan Wajah" card for users without face data
   - Shows face registration status

2. **Face Registration Flow:**

   - User-friendly 3-step wizard
   - Clear instructions for each position
   - Real-time face detection feedback
   - Preview before saving

3. **Database Integration:**

   - Stores face encodings for each user
   - Links to user account
   - Supports multiple angles for better recognition

4. **Attendance System Ready:**

   - Users with registered faces can use face attendance
   - Compares live face with stored data
   - Accurate recognition from different angles

5. **Admin Access Fixed:**
   - Admin can login with admin/admin
   - Full access to system features

## TESTING CHECKLIST:

- [ ] Admin can login successfully
- [ ] Dashboard shows face registration status
- [ ] Face registration wizard works smoothly
- [ ] All 3 face positions captured correctly
- [ ] Face data saved to database
- [ ] Registered users can use face attendance
- [ ] Face matching
