from flask import Flask, request, jsonify
import main  # Import your main script here

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Your main script should have a function to process the incoming data
    # and return a prediction. For this example, let's say it's called `process_input`.
    prediction = main.process_input(data)

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
