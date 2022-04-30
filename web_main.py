#import flasks libraries
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import webbrowser
from threading import Timer
from MaskDetection import MaskDetection


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/',  methods=['GET'])
def index():
    return render_template('home.html', title="Home")

def gen(camera):
    while True:
        (frame, jml_masks) = camera.get_mask()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/video_fedd')
def video_feed():
    return Response(gen(MaskDetection()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteksi', methods=['GET','POST'])
def deteksi():
    return render_template('deteksi.html', title="Deteksi")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
    #Timer(1,lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    #socketio.run(app)
