from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)

# Initialize and configure Picamera2 for video streaming
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2) # Warm-up sensor

def generate_frames():
    while True:
        # Capture frame as a numpy array
        frame = picam2.capture_array()
        
        # Convert RGB888 to BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame_bgr)
        if not success:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pi Zero Live Stream</title>
            <style>
                body { background-color: #121212; color: #ffffff; font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                img { border: 4px solid #333; border-radius: 8px; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); }
            </style>
        </head>
        <body>
            <h2>Raspberry Pi Zero Live Camera Feed</h2>
            <img src="{{ url_for('video_feed') }}" width="640" height="480" />
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run server accessible to your local network on port 5000
    app.run(host='0.0.0.0', port=5000, threaded=True)
