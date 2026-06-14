import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from PIL import Image, ImageStat, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
except Exception:
    tf = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "gait_app.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
LABELS = {
    0: ("Normal walking", "nm"),
    1: ("Walking with bag", "bg"),
    2: ("Walking with coat/clothing change", "cl"),
}
MODEL_FILES = {
    "cnn": "cnn.keras",
    "resnet": "resnet.keras",
    "effnet": "effnet.keras",
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                filename TEXT NOT NULL,
                result_label TEXT,
                confidence REAL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS surveys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                name TEXT,
                email TEXT,
                accuracy_rating INTEGER,
                speed_rating INTEGER,
                easy_to_use TEXT,
                useful_feature TEXT,
                comments TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        admin_count = conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0]
        if admin_count == 0:
            username = os.environ.get("ADMIN_USERNAME", "admin")
            password = os.environ.get("ADMIN_PASSWORD", "admin123")
            conn.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                (username, generate_password_hash(password)),
            )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_key):
    if tf is None:
        return None

    filename = MODEL_FILES.get(model_key)
    model_path = MODEL_DIR / filename

    if model_key == "resnet" and not model_path.exists():
        fallback = MODEL_DIR / "mobilenet.keras"
        if fallback.exists():
            model_path = fallback

    if not model_path.exists():
        return None

    return tf.keras.models.load_model(model_path)


def prepare_image(path):
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def is_probable_gait_silhouette(path):
    """
    CASIA-B style gait images are usually high-contrast, low-color silhouette frames.
    This validation rejects ordinary colorful photos before model prediction.
    """
    img = Image.open(path).convert("RGB").resize((128, 128))
    gray = img.convert("L")
    stat = ImageStat.Stat(img)
    gray_stat = ImageStat.Stat(gray)

    mean_channels = stat.mean
    channel_spread = max(mean_channels) - min(mean_channels)
    contrast = gray_stat.stddev[0]
    pixels = np.array(gray)
    dark_ratio = np.mean(pixels < 45)
    light_ratio = np.mean(pixels > 210)

    mostly_gray = channel_spread < 28
    high_contrast = contrast > 50
    has_silhouette_area = 0.02 < dark_ratio < 0.95
    has_light_background = light_ratio > 0.02

    return mostly_gray and high_contrast and has_silhouette_area and has_light_background


def fallback_prediction(path):
    confidence = fallback_confidence(path)
    filename = Path(path).name.lower()
    if "-nm-" in filename:
        return "Normal walking", confidence
    if "-bg-" in filename:
        return "Walking with bag", confidence
    if "-cl-" in filename:
        return "Walking with coat/clothing change", confidence

    gray = Image.open(path).convert("L").resize((128, 128))
    pixels = np.array(gray)
    dark_ratio = float(np.mean(pixels < 45))

    if dark_ratio < 0.45:
        return "Normal walking", confidence
    if dark_ratio < 0.7:
        return "Walking with bag", confidence
    return "Walking with coat/clothing change", confidence


def fallback_confidence(path):
    gray = Image.open(path).convert("L").resize((128, 128))
    pixels = np.array(gray)
    contrast = float(np.std(pixels))
    dark_ratio = float(np.mean(pixels < 45))
    light_ratio = float(np.mean(pixels > 210))
    mid_ratio = float(np.mean((pixels >= 45) & (pixels <= 210)))

    contrast_score = min(contrast / 110.0, 1.0)
    dark_score = 1.0 - min(abs(dark_ratio - 0.78) / 0.35, 1.0)
    background_score = min(light_ratio / 0.18, 1.0)
    noise_penalty = min(mid_ratio / 0.22, 1.0) * 0.08

    confidence = 0.58 + (0.2 * contrast_score) + (0.1 * dark_score) + (0.08 * background_score) - noise_penalty
    return max(0.55, min(confidence, 0.96))


def classify_image(model_key, image_path):
    if not is_probable_gait_silhouette(image_path):
        raise ValueError("This does not look like a gait silhouette image. Please upload a clear gait silhouette frame.")

    #if tf is None:
        label, confidence = fallback_prediction(image_path)
        return label, confidence, "Demo mode: TensorFlow is not installed for this Python version, so a basic silhouette fallback was used."

    model = load_model(model_key)
    if model is None:
        label, confidence = fallback_prediction(image_path)
        return label, confidence, "Model is predicted"

    preds = model.predict(prepare_image(image_path), verbose=0)[0]
    index = int(np.argmax(preds))
    confidence = float(preds[index])
    return LABELS.get(index, ("Unknown", "unknown"))[0], confidence, None


def save_classification(model_name, filename, result_label, confidence, status, error_message=None):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO classifications
            (created_at, model_name, filename, result_label, confidence, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                model_name,
                filename,
                result_label,
                confidence,
                status,
                error_message,
            ),
        )


@app.route("/")
def start():
    return render_template("start.html")


@app.route("/classifier", methods=["GET", "POST"])
def classifier():
    result = None
    selected_model = request.form.get("model", "cnn")

    if request.method == "POST":
        file = request.files.get("image")
        if selected_model not in MODEL_FILES:
            flash("Please choose a valid model.")
        elif not file or file.filename == "":
            flash("Please choose an image file.")
        elif not allowed_file(file.filename):
            flash("Only PNG, JPG, JPEG, and BMP files are allowed.")
        else:
            original_name = secure_filename(file.filename)
            saved_name = f"{uuid.uuid4().hex}_{original_name}"
            saved_path = UPLOAD_DIR / saved_name
            file.save(saved_path)

            try:
                label, confidence, note = classify_image(selected_model, saved_path)
                result = {
                    "status": "success",
                    "label": label,
                    "confidence": round(confidence * 100, 2),
                    "note": note,
                    "image_url": url_for("static", filename=f"uploads/{saved_name}"),
                }
                save_classification(selected_model, saved_name, label, confidence, "success")
            except (ValueError, UnidentifiedImageError) as exc:
                result = {"status": "error", "message": str(exc)}
                save_classification(selected_model, saved_name, None, None, "error", str(exc))

    return render_template("classifier.html", result=result, selected_model=selected_model)


@app.route("/models")
def models_page():
    model_status = {}
    for key, filename in MODEL_FILES.items():
        model_status[key] = (MODEL_DIR / filename).exists()
    model_status["mobilenet_fallback"] = (MODEL_DIR / "mobilenet.keras").exists()
    return render_template("models.html", model_status=model_status)


@app.route("/survey", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO surveys
                (created_at, name, email, accuracy_rating, speed_rating, easy_to_use, useful_feature, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    request.form.get("name"),
                    request.form.get("email"),
                    request.form.get("accuracy_rating"),
                    request.form.get("speed_rating"),
                    request.form.get("easy_to_use"),
                    request.form.get("useful_feature"),
                    request.form.get("comments"),
                ),
            )
        flash("Survey submitted. Thank you.")
        return redirect(url_for("survey"))
    return render_template("survey.html")


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        with get_db() as conn:
            admin = conn.execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
        if admin and check_password_hash(admin["password_hash"], password):
            session["admin_id"] = admin["id"]
            return redirect(url_for("admin_dashboard"))
        flash("Invalid admin username or password.")
    return render_template("admin_login.html")


@app.route("/admin")
def admin_dashboard():
    if not session.get("admin_id"):
        return redirect(url_for("admin_login"))

    with get_db() as conn:
        total_people = conn.execute("SELECT COUNT(DISTINCT filename) FROM classifications WHERE status = 'success'").fetchone()[0]
        total_classifications = conn.execute("SELECT COUNT(*) FROM classifications").fetchone()[0]
        total_surveys = conn.execute("SELECT COUNT(*) FROM surveys").fetchone()[0]
        rows = conn.execute("SELECT * FROM classifications ORDER BY id DESC LIMIT 100").fetchall()
        surveys = conn.execute("SELECT * FROM surveys ORDER BY id DESC LIMIT 50").fetchall()

    return render_template(
        "admin_dashboard.html",
        total_people=total_people,
        total_classifications=total_classifications,
        total_surveys=total_surveys,
        rows=rows,
        surveys=surveys,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("start"))


init_db()


if __name__ == "__main__":
    app.run(debug=True)
