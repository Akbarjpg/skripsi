# Agent Prompt: Fix Routing and 404 Errors in Face Recognition Attendance System

## Current Issue
Saya mengalami masalah routing di aplikasi web Face Recognition Attendance System. Ketika mencoba mengakses halaman login atau memulai demo absensi, saya mendapatkan error:
- `"GET /login HTTP/1.1" 404 -`
- `"GET /.well-known/appspecific/com.chrome.devtools.json HTTP/1.1" 404 -`

## Project Structure
```
dari nol/
├── src/
│   ├── web/
│   │   ├── app.py              # Main Flask application
│   │   ├── templates/
│   │   │   ├── index.html      # Home page
│   │   │   └── attendance.html # Attendance page
│   │   └── static/
│   │       ├── css/
│   │       └── js/
│   ├── models/
│   │   ├── cnn_model.py        # CNN for liveness detection
│   │   └── landmark_detector.py # Facial landmark detection
│   ├── challenges/
│   │   └── challenge_response.py # Challenge-response system
│   └── utils/
│       └── fusion.py           # Fusion algorithm
├── test_img/                   # Dataset folder
│   └── color/                  # RGB images
├── launch.py                   # Launch script
├── fallback_app.py            # Fallback Flask app
└── requirements.txt           # Dependencies
```

## Routes yang Seharusnya Ada
1. `/` - Home page (index.html)
2. `/login` - Login page
3. `/register` - Registration page
4. `/attendance` - Attendance verification page
5. `/api/verify` - API endpoint for face verification
6. `/api/enroll` - API endpoint for face enrollment
7. `/api/challenge` - API endpoint for getting challenges
8. `/logout` - Logout functionality

## Specific Tasks to Fix

### 1. Route Definition Issues
- Periksa dan pastikan semua routes sudah didefinisikan dengan benar di `app.py`
- Tambahkan route `/login` yang missing
- Implementasi proper redirect logic
- Handle static file serving correctly

### 2. Template Issues
- Buat template `login.html` yang missing
- Pastikan semua template paths benar
- Verifikasi template inheritance jika ada

### 3. Session Management
- Implementasi proper session handling
- Login/logout functionality
- User authentication flow

### 4. API Endpoints
- Pastikan semua API endpoints return proper JSON responses
- Handle CORS if needed
- Proper error handling dengan status codes yang benar

### 5. Static Files
- Configure static file serving correctly
- Ensure CSS/JS files are accessible
- Fix any broken asset links

## Code Fixes Needed

### Fix 1: Add Missing Routes in app.py
```python
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic
        pass
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle registration logic
        pass
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
```

### Fix 2: Create Missing Templates
Create `login.html` and `register.html` templates with proper forms and Bootstrap styling.

### Fix 3: Update Navigation Flow
- Home page → Login/Register → Attendance
- Implement proper authentication checks
- Redirect unauthenticated users to login

### Fix 4: Error Handling
Add proper 404 error handler:
```python
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
```

## Testing Requirements
1. Test all routes return correct status codes
2. Verify template rendering works
3. Check session management functionality
4. Test API endpoints with proper payloads
5. Ensure static files load correctly

## Expected Outcome
- All routes should work without 404 errors
- Proper navigation flow from login to attendance
- Working face verification system
- No console errors in browser
- Smooth user experience

## Additional Improvements
1. Add proper logging for debugging
2. Implement database for user storage
3. Add password hashing for security
4. Create admin dashboard
5. Add attendance reports functionality

Please help me fix these routing issues step by step, starting with the missing `/login` route and ensuring all components work together seamlessly.