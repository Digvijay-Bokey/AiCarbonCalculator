from flask import Flask, request, jsonify
import main  # Import your main script here

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    prediction = main.process_input(data)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
