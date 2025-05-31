import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from flask_wtf.recaptcha import RecaptchaField
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import numpy as np
import pillow_heif
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'secret123')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RECAPTCHA_PUBLIC_KEY'] = os.environ.get('RECAPTCHA_PUBLIC_KEY')
app.config['RECAPTCHA_PRIVATE_KEY'] = os.environ.get('RECAPTCHA_PRIVATE_KEY')
app.config['RECAPTCHA_TYPE'] = os.environ.get('RECAPTCHA_TYPE', 'invisible')

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

app.config['SESSION_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_SECURE'] = True

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

model = load_model('trained_model.h5')
print("✅ Model loaded from trained_model.h5")
IMG_HEIGHT, IMG_WIDTH = 128, 128


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    predictions = db.relationship('ImagePrediction', backref='user', lazy=True)


class ImagePrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    note = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    recaptcha = RecaptchaField()
    submit = SubmitField('Register')


class ResendForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    recaptcha = RecaptchaField()
    submit = SubmitField('Resend Verification')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='demo_user').first():
        demo_user = User(
            username='demo_user',
            email='demo@example.com',
            password_hash=generate_password_hash('demopassword'),
            is_verified=True
        )
        db.session.add(demo_user)
        db.session.commit()


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = 'Fake (AI-Generated)' if prediction < 0.5 else 'Real'
    confidence = 1 - \
        float(prediction) if prediction < 0.5 else float(prediction)
    return label, round(confidence * 100, 2)


@app.route('/demo-login')
def demo_login():
    user = User.query.filter_by(username='demo_user').first()
    if user:
        login_user(user)
        flash('Logged in as demo user.', 'success')
    else:
        flash('Demo user account not found.', 'danger')
    return redirect(url_for('home'))


@app.route('/healthcheck')
def healthcheck():
    return "OK", 200


@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        label, confidence = predict_image(filepath)
        new_prediction = ImagePrediction(
            user_id=current_user.id,
            image_filename=filename,
            result=label,
            confidence=confidence
        )
        db.session.add(new_prediction)
        db.session.commit()
        return render_template('result.html', label=label, confidence=confidence, image_url=filepath)
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('User or email already exists.')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email,
                        password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        token = serializer.dumps(email, salt='email-confirm')
        link = url_for('confirm_email', token=token, _external=True)
        msg = Message('Confirm your Email for the Image Verification App', sender=os.environ.get(
            'MAIL_USERNAME'), recipients=[email])
        msg.html = f"<h3>Confirm Your Email</h3><p><a href='{link}'>Verify Email</a></p>"
        mail.send(msg)
        flash('Registration successful. Check your email to confirm.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=3600)
    except SignatureExpired:
        return '<h1>Token expired!</h1>'
    user = User.query.filter_by(email=email).first_or_404()
    if not user.is_verified:
        user.is_verified = True
        db.session.commit()
        return '<h1>Email confirmed. You may now <a href="/login">log in</a>.</h1>'
    return '<h1>Account already verified.</h1>'


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            if not user.is_verified:
                flash('Please verify your email first.')
                return redirect(url_for('login'))
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials.')
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    if current_user.username == 'demo_user':
        # Delete demo user's predictions
        ImagePrediction.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
    logout_user()
    return redirect(url_for('home'))


@app.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    form = ResendForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if not user:
            flash('Email not found.')
            return redirect(url_for('resend_verification'))
        if user.is_verified:
            flash('Account already verified.')
            return redirect(url_for('login'))
        token = serializer.dumps(email, salt='email-confirm')
        link = url_for('confirm_email', token=token, _external=True)
        msg = Message('Resend: Confirm your Email', sender=os.environ.get(
            'MAIL_USERNAME'), recipients=[email])
        msg.html = f"<h3>Confirm Your Email</h3><p><a href='{link}'>Verify Email</a></p>"
        mail.send(msg)
        flash('Verification email resent.')
        return redirect(url_for('login'))
    return render_template('resend_verification.html', form=form)


@app.route('/history')
@login_required
def history():
    predictions = ImagePrediction.query.filter_by(
        user_id=current_user.id).all()
    return render_template('history.html', predictions=predictions)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():

    sample_images = []
    sample_dir = os.path.join('static', 'sample_images')
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(
            sample_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if request.method == 'POST':
        file = request.files.get('image')
        sample_image = request.form.get('sample_image')
        note = request.form.get('note', '')

        # Use uploaded file or selected sample image
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        elif sample_image:
            filename = secure_filename(sample_image)
            src_path = os.path.join('static', 'sample_images', filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.open(src_path).save(filepath)  # Copy image
        else:
            flash("Please upload an image or select a sample.", "danger")
            return redirect(url_for('upload'))

        label, confidence = predict_image(filepath)

        new_pred = ImagePrediction(
            user_id=current_user.id,
            image_filename=filename,
            result=label,
            confidence=confidence,
            note=note
        )
        db.session.add(new_pred)
        db.session.commit()

        return render_template('result.html', label=label, confidence=confidence, image_url=filepath)

    return render_template('upload.html', sample_images=sample_images)


@app.route('/upload-sample', methods=['POST'])
@login_required
def upload_sample():
    filename = request.form['filename']
    filepath = os.path.join('static/sample_images', filename)

    label, confidence = predict_image(filepath)

    new_pred = ImagePrediction(
        user_id=current_user.id,
        image_filename=filename,
        result=label,
        confidence=confidence
    )
    db.session.add(new_pred)
    db.session.commit()

    return render_template('result.html', label=label, confidence=confidence, image_url=filepath)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)))
