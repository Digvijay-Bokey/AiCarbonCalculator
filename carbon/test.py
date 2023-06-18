from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def generate_sentence():
    return render_template('data_input.html')

@app.route('/results')


if __name__ == '__main__':
    app.run(debug=True)
