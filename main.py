import json

from flask import Flask, render_template, request, jsonify, send_file
from core.recognition import process_json, process_json_test, process_file
from flask_socketio import emit
from flask_socketio import SocketIO

app = Flask(__name__)
# socketio = SocketIO(app)
#
# thread = None
#
#
# def progress_update():
#     for i in range(5):
#         socketio.sleep(1)
#         socketio.emit("progress", {"text": 25 * i})
#         print("progress")
#
#
# @socketio.on("connect")
# def test_connect():
#     print("connect")
#     socketio.start_background_task(target=progress_update)
#
#
# @socketio.on("disconnect")
# def test_disconnect():
#     print("disconnect")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        return send_file(process_file(uploaded_file))
    else:
        return render_template('index.html')


@app.route('/api/v1.0/get_text', methods=['POST'])
def get_text():
    if not request.json:
        pass
    return process_json(request.json)


@app.route('/api/v1.0/test', methods=['GET'])
def get_text_test():
    if not request.json:
        pass
    response = app.response_class(
        response=json.dumps(process_json_test(request.json)),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    # socketio.run(app, host='10.77.144.64', port=8000, debug=True)
    app.run(debug=True, host='10.77.144.64', port=8000)
1