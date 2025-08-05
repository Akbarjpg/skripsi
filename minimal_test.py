"""
Minimal working Flask app for testing
"""

from flask import Flask, render_template
import os

# Get the correct template directory
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'src', 'web', 'templates')
static_dir = os.path.join(current_dir, 'src', 'web', 'static')

print(f"Template directory: {template_dir}")
print(f"Template directory exists: {os.path.exists(template_dir)}")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/test')
def test():
    return "App is working!"

if __name__ == '__main__':
    print("Starting minimal test app...")
    print(f"Templates in: {template_dir}")
    if os.path.exists(template_dir):
        files = os.listdir(template_dir)
        print(f"Available templates: {files}")
    
    app.run(debug=True, port=5001)
