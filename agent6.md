# Prompt untuk Memperbaiki Error JSON dan Landmark Detection yang Lambat

Program anti-spoofing saya mengalami masalah serius setelah optimasi:

## 1. ERROR CRITICAL:

```
Error: Processing error: Object of type bool is not JSON serializable
```

Error ini muncul berulang-ulang dan menyebabkan disconnected dari server.

## 2. LANDMARK DETECTION SANGAT LAMBAT:

- Menunggu sampai 3 MENIT untuk landmark muncul
- Titik landmark TIDAK BERGERAK sama sekali
- Tidak mengikuti gerakan wajah
- Sistem seperti freeze/hang

## ANALISIS YANG DIPERLUKAN:

### 1. Fix JSON Serialization Error

- Periksa semua data yang dikirim melalui SocketIO
- Cari nilai boolean yang tidak dikonversi properly
- Pastikan semua numpy types dikonversi ke Python native types
- Check di bagian `emit()` dan `json.dumps()`

### 2. Fix Landmark Detection Performance

- Landmark detection mungkin blocking main thread
- MediaPipe mungkin tidak initialized properly
- Frame processing mungkin stuck di queue
- GPU/CPU resource mungkin bottleneck

### 3. Possible Root Causes

- Threading issue causing deadlock
- Memory leak causing system slowdown
- Cache system malfunction
- Frame skip logic error
- MediaPipe model loading issue

## SOLUSI YANG DIPERLUKAN:

### 1. Immediate Fixes untuk JSON Error:

```python
# Convert all numpy/bool types before sending
def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
```

### 2. Fix Landmark Detection:

- Remove/reduce frame skipping
- Check MediaPipe initialization
- Add timeout to prevent hanging
- Use simpler landmark model
- Add error recovery mechanism

### 3. Performance Debug:

- Add detailed timing logs
- Monitor CPU/GPU usage
- Check memory consumption
- Trace where the delay happens

## OUTPUT YANG DIHARAPKAN:

1. **JSON Error Fixed:**

   - Tidak ada error serialization
   - Koneksi stabil ke server
   - Data flow lancar

2. **Landmark Detection Fixed:**

   - Landmark muncul dalam < 1 detik
   - Titik mengikuti wajah real-time
   - Smooth tracking tanpa lag
   - FPS minimal 15-20

3. **Debug Information:**
   - Log timing untuk setiap tahap processing
   - Memory usage tracking
   - Error recovery mechanism

## CONSTRAINTS:

- Tetap gunakan 3 metode keamanan
- Jangan sacrifice accuracy
- Real-time performance adalah prioritas
- Code harus stable dan production-ready

Tolong perbaiki kedua masalah ini dengan fokus pada:

1. Konversi data types yang benar untuk JSON
2. Non-blocking landmark detection
3. Proper error handling dan
