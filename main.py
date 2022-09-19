from flask import Flask, render_template, request
from core.recognition import process_json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')ффвфы


@app.route('/api/v1.0/get_text', methods=['POST'])
def get_text():
    if not request.json:
        pass
    return process_json(request.json)


if __name__ == '__main__':
    app.run(debug=True)
