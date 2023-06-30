from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

# Init app
app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/carbon_impact'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Init db
db = SQLAlchemy(app)

# Init ma
ma = Marshmallow(app)

# User Class/Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(100))

    def __init__(self, region):
        self.region = region

# User Schema
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        sqla_session = db.session

# Init schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # assuming there's an input field with name="region" in your form
        region = request.form['region']

        new_user = User(region)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('home'))  # Redirect to home after post
    else:
        # Handle GET request here
        return render_template('index.html')

# Run server
if __name__ == '__main__':
    app.run(debug=True)
