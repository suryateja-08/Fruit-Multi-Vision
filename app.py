from flask import Flask,render_template,request,redirect,session,url_for
import csv
import os
import classify
from PIL import Image
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import bcrypt
from datetime import datetime

app = Flask(__name__)
port = int(os.getenv('PORT', 5000))
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///Model.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '1A2bc4s'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(128),unique=True, nullable = False)
    email = db.Column(db.String(128),unique=True, nullable = False)
    password = db.Column(db.String(128), nullable = False)
    
    def __repr__(self) -> str:
        return  f'<User {self.username}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

class History(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    userId = db.Column(db.Integer,db.ForeignKey('user.id'))
    image = db.Column(db.String(128),nullable=False)
    result = db.Column(db.String(128),nullable=False)
    date_created = db.Column(db.DateTime,default=datetime.utcnow)

    def __repr__(self) -> str:
        return  f'<Result {self.result} - {self.userId}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'userId': self.userId,
            'image': self.image,
            'result': self.result,
            'date_created': self.date_created
        }
with app.app_context():
    db.create_all()
    
def getUser():
    userId = session['userId']
    data = User.query.get(userId)
    details = data.to_dict()
    data = History.query.filter(History.userId == userId).order_by(History.date_created.desc()).all()
    history = []
    for data in data:
        history.append(data.to_dict())
    
    return details,history

#class names for ML prediction
class_names =['abiu', 'acai', 'acerola', 'ackee', 'ambarella', 'barberry', 'camu_camu', 'cempedak', 'cherimoya', 'chico', 'cluster_fig', 'cupuacu', 'damson', 'feijoa', 'finger_lime', 'goumi', 'hawthorn', 'hog_plum', 'jaboticaba', 'jambul', 'jocote', 'kaffir_lime', 'kumquat', 'lablab', 'langsat', 'mabolo', 'mangosteen', 'medlar', 'mock_strawberry', 'morinda', 'oil_palm', 'otaheite_apple', 'pawpaw', 'prikly_pear', 'quince', 'redcurrant', 'salak', 'taxus_baccata', 'ugli_fruit', 'white_currant', 'yali_pear']

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template("model.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/result')
def result():
    return render_template("result_train.html")

@app.route('/dataset')
def dataset():
    return render_template("dataset.html")

@app.route('/logout')
def logout():
    session['logged'] = False
    session.pop('userId',None)
    return redirect('/')

@app.route('/update', methods=['POST'])
def update():
    userId = session['userId']
    username = request.form.get('name')
    email = request.form.get('email')
    user = User.query.get(userId)
    user.username = username
    user.email = email
    db.session.commit()
    return redirect('/profile')

@app.route('/changePassword',methods=['POST'])
def changePassword():
    userId = session['userId']
    oldPass = request.form.get('oldPass')
    newPass = request.form.get('newPass')
    confPass = request.form.get('confPass')

    user = User.query.filter_by(id=userId).first()
    details,history = getUser()    
    if user:
        password_match = bcrypt.checkpw(oldPass.encode('utf-8'),user.password)
        if password_match:
            if newPass == confPass:
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(newPass.encode('utf-8'),salt)
                user.password = password_hash
                db.session.commit()
                return redirect('/profile')
            else:
                return render_template('profile.html',details=details,history_items=history,newPassError="New Passwords do not match")
        else:
            return render_template('profile.html',details=details,history_items=history,oldPassError="Old Password is Incorrect")
    else:
        return redirect('/login')

@app.route('/profile')
def profile():
    details,history = getUser()    
    return render_template('profile.html',details=details,history_items=history)
        
@app.route('/output',methods=['POST','GET'])
def output():
    image_path = ''
    if request.method=='POST':
        image_file = request.files["imagefile"]
        modelNo = int(request.form.get('model'))
        
        #saving image in local storage for rendering preview
        image_file.save(f'static/images/test/{image_file.filename}')
        image_path = "static/images/test/" + image_file.filename
        result =''
        image = Image.open(image_file)
        prediction = classify.predict(image,modelNo)
        result = " {} with a {:.2f}% Confidence.".format(class_names[np.argmax(prediction)], 100 * np.max(prediction))
        userId = session['userId']
        new_history = History(userId=userId,result=result,image=image_path)
        db.session.add(new_history)
        db.session.commit()
        print(new_history)
    else:
        result=None
    
    #return result and image in production mode
    return render_template("output.html",result=result,image=image_path)

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('name')
        email = request.form.get('email')
        newPass = request.form.get('newPass')
        confPass = request.form.get('confPass')
                
        if not username or not confPass or not newPass or not email:
            return render_template('login.html',error="Please enter all required fields")
        
        if newPass == confPass:
            try:
                salt = bcrypt.gensalt()
                password_hash = bcrypt.hashpw(confPass.encode('utf-8'),salt)
                new_user= User(username=username,email=email,password=password_hash)
                db.session.add(new_user)
                db.session.commit()
                return redirect('/login')
        
            except IntegrityError:
                return render_template('login.html',error="username or email alredy exists")
        else:
            return render_template('login.html',error="Passwords do not match")
        
@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return render_template('login.html',error="Send all the required data")
                
        user = User.query.filter_by(username=username).first()
        print(user)
        if not user:
            return render_template('login.html',error="No User Found")
        
        password_match = bcrypt.checkpw(password.encode('utf-8'),user.password)
        if password_match:
            session['logged'] = True
            session['userId'] = user.id
            print(user.id)
            return redirect('/output')
        else:
            return render_template('login.html',error="Password Incorrect")


if __name__ == '__main__':
    app.run(debug=True,port=port)