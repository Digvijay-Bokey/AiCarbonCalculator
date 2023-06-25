from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def generate_sentence():
    return render_template('data_input.html')

if __name__ == '__main__':
    app.run(debug=True)
