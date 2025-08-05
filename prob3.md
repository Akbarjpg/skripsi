# Prompt untuk Memperbaiki Error "Terjadi kesalahan saat memproses gambar" di Face Registration

## MASALAH UTAMA:

Setiap kali mencoba mengambil foto di halaman `http://localhost:5000/register-face`, selalu muncul error message "Terjadi kesalahan saat memproses gambar". Ini terjadi di semua step (front, left, right).

## GEJALA:

1. Camera/webcam berfungsi normal - video stream terlihat
2. Countdown 3-2-1 berjalan dengan baik
3. Saat capture foto, selalu error dengan pesan yang sama
4. Tidak bisa melanjutkan ke step berikutnya

## ANALISIS YANG DIPERLUKAN:

### 1. Check SocketIO Connection

- Apakah koneksi SocketIO berhasil established?
- Apakah event 'capture_face' terkirim dengan benar?
- Apakah server menerima data image?

### 2. Check Image Data Format

```javascript
// Di frontend, check format image data yang dikirim
console.log("Image data type:", typeof imageData);
console.log("Image data prefix:", imageData.substring(0, 50));
console.log("Image data length:", imageData.length);
```

### 3. Check Server-Side Processing

```python
@socketio.on('capture_face')
def handle_capture_face(data):
    print(f"Received capture_face event")
    print(f"Data keys: {data.keys()}")
    print(f"Position: {data.get('position')}")
    print(f"Image data length: {len(data.get('image', ''))}")
    print(f"Session ID: {session.get('user_id')}")
```

### 4. Common Issues to Check:

#### A. Base64 Encoding Issue

- Image data mungkin tidak ter-encode dengan benar
- Format base64 string mungkin salah (missing prefix "data:image/jpeg;base64,")

#### B. Face Recognition Library Error

- Library face_recognition mungkin tidak terinstall
- Error saat extract face encodings
- No face detected in image

#### C. Database Error

- Error saat save ke database
- User session tidak valid
- Foreign key constraint

#### D. Image Size/Quality

- Image terlalu besar
- Image quality terlalu rendah
- Canvas dimensions issue

## DEBUGGING STEPS:

### 1. Add Comprehensive Logging

```python
@socketio.on('capture_face')
def handle_capture_face(data):
    try:
        print("=== CAPTURE FACE DEBUG ===")

        # 1. Check basic data
        if not data:
            emit('capture_result', {
                'success': False,
                'message': 'No data received'
            })
            return

        position = data.get('position')
        image_data = data.get('image')

        print(f"Position: {position}")
        print(f"Image data exists: {bool(image_data)}")

        # 2. Check session
        user_id = session.get('user_id')
        print(f"User ID from session: {user_id}")

        if not user_id:
            emit('capture_result', {
                'success': False,
                'message': 'User not logged in'
            })
            return

        # 3. Process base64 image
        try:
            # Remove data URL prefix if exists
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(image_data)
            print(f"Decoded image size: {len(image_bytes)} bytes")

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            print(f"Image shape: {image.shape}")

        except Exception as e:
            print(f"Image processing error: {str(e)}")
            emit('capture_result', {
                'success': False,
                'message': 'Failed to process image data'
            })
            return

        # 4. Face detection
        try:
            if face_recognition:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces
                face_locations = face_recognition.face_locations(rgb_image)
                print(f"Faces found: {len(face_locations)}")

                if not face_locations:
                    emit('capture_result', {
                        'success': False,
                        'message': 'No face detected in image'
                    })
                    return

                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                if not face_encodings:
                    emit('capture_result', {
                        'success': False,
                        'message': 'Could not extract face features'
                    })
                    return

                face_encoding = face_encodings[0]
                print(f"Face encoding shape: {face_encoding.shape}")

            else:
                # Fallback if face_recognition not available
                print("Using fallback face detection")
                face_encoding = np.random.rand(128).tolist()

        except Exception as e:
            print(f"Face detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            emit('capture_result', {
                'success': False,
                'message': f'Face detection failed: {str(e)}'
            })
            return

        # 5. Save to database
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()

            # Check if face already exists for this position
            cursor.execute('''
                DELETE FROM face_data
                WHERE user_id = ? AND face_position = ?
            ''', (user_id, position))

            # Save face encoding
            cursor.execute('''
                INSERT INTO face_data (user_id, face_position, face_encoding)
                VALUES (?, ?, ?)
            ''', (user_id, position, json.dumps(face_encoding.tolist())))

            conn.commit()
            conn.close()

            print(f"Successfully saved face data for position: {position}")

            emit('capture_result', {
                'success': True,
                'message': f'Foto {position} berhasil disimpan',
                'position': position
            })

        except Exception as e:
            print(f"Database error: {str(e)}")
            emit('capture_result', {
                'success': False,
                'message': 'Failed to save face data'
            })

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('capture_result', {
            'success': False,
            'message': 'Terjadi kesalahan saat memproses gambar'
        })
```

### 2. Update Frontend Error Handling

```javascript
handleCaptureResult(data) {
    console.log('Capture result:', data);

    if (data.success) {
        // Success handling
        this.updateStatus(`✓ ${data.message}`, 'success');
    } else {
        // Show specific error message
        console.error('Capture failed:', data.message);
        this.updateStatus(data.message || 'Terjadi kesalahan saat memproses gambar', 'error');

        // Re-enable capture after error
        setTimeout(() => {
            this.isCapturing = false;
            this.startFaceDetection();
        }, 3000);
    }
}
```

### 3. Test Alternative Image Format

```javascript
capturePhoto() {
    // Try different image quality/format
    const quality = 0.8; // Reduce quality to reduce size
    const imageData = this.canvas.toDataURL('image/jpeg', quality);

    console.log('Capturing photo with quality:', quality);
    console.log('Image data size:', imageData.length);

    // Send to server
    socket.emit('capture_face', {
        position: this.currentPosition,
        image: imageData
    });
}
```

## POSSIBLE SOLUTIONS:

### 1. Install Missing Dependencies

```bash
pip install face-recognition
pip install opencv-python
pip install numpy
```

### 2. Add Fallback for Missing Libraries

```python
# In app_optimized.py
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not installed, using mock data")
```

### 3. Simplify Face Registration (Temporary)

If face_recognition is problematic, temporarily store just the images:

```python
# Save image file instead of encoding
image_filename = f"face_{user_id}_{position}_{int(time.time())}.jpg"
image_path = os.path.join('static', 'faces', image_filename)

# Create directory if not exists
os.makedirs(os.path.dirname(image_path), exist_ok=True)

# Save image
with open(image_path, 'wb') as f:
    f.write(image_bytes)

# Store path in database
cursor.execute('''
    INSERT INTO face_data (user_id, face_position, image_path)
    VALUES (?, ?, ?)
''', (user_id, position, image_path))
```

## EXPECTED OUTCOME:

1. Detailed error messages showing exactly where the process fails
2. Successful photo capture and storage
3. Ability to complete all 3 steps of face registration
4. Clear feedback to user about what went wrong if errors occur

## TESTING CHECKLIST:

- [ ] Check browser console for JavaScript errors
- [ ] Check server console for Python errors
- [ ] Verify face_recognition library is installed
- [ ] Test with different browsers
- [ ] Check database permissions
- [ ] Verify user is logged in with valid session
-
