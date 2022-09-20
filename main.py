import json

from flask import Flask, render_template, request, jsonify, send_file
from core.recognition import process_json, process_json_test

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        return send_file(process_json_test(uploaded_file.filename))
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
    app.run(debug=True, host='10.77.144.64', port=8000)
