from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os

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

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Create a User
@app.route('/user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        region = request.json['region']

        new_user = User(region)

        db.session.add(new_user)
        db.session.commit()

        return user_schema.jsonify(new_user)
    elif request.method == 'GET':
        # Handle GET request here
        return render_template('user.html')


# Run server
if __name__ == '__main__':
    app.run(debug=True)
