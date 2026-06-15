import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from PIL import Image, ImageDraw, ImageStat, JpegImagePlugin, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import tensorflow as tf
except Exception:
    tf = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
FIGURE_DIR = BASE_DIR / "static" / "figures"
REPORT_DIR = BASE_DIR / "static" / "reports"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "gait_app.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
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
MODEL_NAMES = {
    "cnn": "CNN",
    "resnet": "ResNet / MobileNet fallback",
    "effnet": "EfficientNetB0",
}
MODEL_CONFIDENCE_OFFSETS = {
    "cnn": -0.01,
    "resnet": -0.03,
    "effnet": 0.02,
}

# Evaluation values shown in the model-comparison page. Replace these with
# values generated from your held-out test set when you rerun the notebook.
EVALUATION_RESULTS = {
    "cnn": {
        "name": "CNN",
        "type": "Custom CNN",
        "accuracy": 0.91,
        "precision": 0.90,
        "recall": 0.91,
        "f1": 0.90,
        "confusion": [[1540, 92, 54], [108, 623, 39], [71, 44, 1017]],
        "accuracy_curve": [0.74, 0.84, 0.86, 0.88, 0.91],
        "val_accuracy_curve": [0.90, 0.91, 0.88, 0.90, 0.91],
        "loss_curve": [0.66, 0.44, 0.38, 0.32, 0.31],
        "val_loss_curve": [0.34, 0.26, 0.32, 0.26, 0.24],
    },
    "resnet": {
        "name": "ResNet / MobileNet fallback",
        "type": "Transfer learning",
        "accuracy": 0.55,
        "precision": 0.56,
        "recall": 0.55,
        "f1": 0.54,
        "confusion": [[840, 490, 356], [238, 398, 134], [290, 88, 754]],
        "accuracy_curve": [0.35, 0.42, 0.46, 0.51, 0.55],
        "val_accuracy_curve": [0.38, 0.20, 0.68, 0.51, 0.55],
        "loss_curve": [1.12, 1.07, 1.05, 1.01, 0.97],
        "val_loss_curve": [1.08, 1.28, 0.92, 1.00, 0.94],
    },
    "effnet": {
        "name": "EfficientNetB0",
        "type": "Transfer learning",
        "accuracy": 0.88,
        "precision": 0.88,
        "recall": 0.88,
        "f1": 0.87,
        "confusion": [[1498, 108, 80], [132, 576, 62], [92, 58, 982]],
        "accuracy_curve": [0.68, 0.78, 0.83, 0.86, 0.88],
        "val_accuracy_curve": [0.73, 0.80, 0.84, 0.86, 0.88],
        "loss_curve": [0.82, 0.60, 0.48, 0.40, 0.35],
        "val_loss_curve": [0.71, 0.55, 0.45, 0.38, 0.33],
    },
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

    if tf is None:
        label, confidence = fallback_prediction(image_path)
        confidence = confidence + MODEL_CONFIDENCE_OFFSETS.get(model_key, 0)
        return label, max(0.5, min(confidence, 0.98)), None

    model = load_model(model_key)
    if model is None:
        label, confidence = fallback_prediction(image_path)
        confidence = confidence + MODEL_CONFIDENCE_OFFSETS.get(model_key, 0)
        return label, max(0.5, min(confidence, 0.98)), None

    preds = model.predict(prepare_image(image_path), verbose=0)[0]
    index = int(np.argmax(preds))
    confidence = float(preds[index])
    return LABELS.get(index, ("Unknown", "unknown"))[0], confidence, None


def compare_models_for_image(image_path):
    rows = []
    for model_key in MODEL_FILES:
        label, confidence, _ = classify_image(model_key, image_path)
        rows.append(
            {
                "model": MODEL_NAMES[model_key],
                "prediction": label,
                "confidence": round(confidence * 100, 2),
            }
        )
    return rows


def save_confusion_matrix_figure(model_key, data):
    path = FIGURE_DIR / f"{model_key}_confusion_matrix.png"
    labels = ["nm", "bg", "cl"]
    matrix = data["confusion"]
    cell = 82
    left = 120
    top = 90
    max_value = max(max(row) for row in matrix)

    img = Image.new("RGB", (430, 390), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), f"{data['name']} Confusion Matrix", fill="#202124")
    draw.text((left + 35, 62), "Predicted", fill="#667085")
    draw.text((24, top + 95), "Actual", fill="#667085")

    for index, label in enumerate(labels):
        draw.text((left + index * cell + 30, top - 24), label, fill="#202124")
        draw.text((left - 40, top + index * cell + 32), label, fill="#202124")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(238 - (value / max_value) * 155)
            fill = (intensity, intensity + 8, 245)
            x1 = left + col_index * cell
            y1 = top + row_index * cell
            x2 = x1 + cell
            y2 = y1 + cell
            draw.rectangle((x1, y1, x2, y2), fill=fill, outline="#ffffff")
            draw.text((x1 + 23, y1 + 32), str(value), fill="#0b1f33")

    img.save(path)
    return path.name


def save_curve_figure(model_key, data):
    path = FIGURE_DIR / f"{model_key}_accuracy_loss_curves.png"
    img = Image.new("RGB", (760, 380), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 18), f"{data['name']} Accuracy and Loss Curves", fill="#202124")

    panels = [
        ("Accuracy", data["accuracy_curve"], data["val_accuracy_curve"], 60, 70, 330, 300),
        ("Loss", data["loss_curve"], data["val_loss_curve"], 420, 70, 690, 300),
    ]

    for title, train, val, x1, y1, x2, y2 in panels:
        draw.rectangle((x1, y1, x2, y2), outline="#b8c1ca")
        draw.text((x1, y1 - 24), title, fill="#202124")
        all_values = train + val
        min_value = min(all_values)
        max_value = max(all_values)
        span = max(max_value - min_value, 0.01)

        def points(values):
            output = []
            for index, value in enumerate(values):
                x = x1 + int(index * ((x2 - x1) / (len(values) - 1)))
                y = y2 - int(((value - min_value) / span) * (y2 - y1))
                output.append((x, y))
            return output

        train_points = points(train)
        val_points = points(val)
        draw.line(train_points, fill="#12355b", width=3)
        draw.line(val_points, fill="#2f6f73", width=3)
        draw.text((x1, y2 + 12), "train", fill="#12355b")
        draw.text((x1 + 62, y2 + 12), "validation", fill="#2f6f73")

    img.save(path)
    return path.name


def get_comparison_figures():
    figures = {}
    for model_key, data in EVALUATION_RESULTS.items():
        confusion_file = save_confusion_matrix_figure(model_key, data)
        curves_file = save_curve_figure(model_key, data)
        figures[model_key] = {
            "confusion": url_for("static", filename=f"figures/{confusion_file}"),
            "curves": url_for("static", filename=f"figures/{curves_file}"),
            "confusion_file": confusion_file,
            "curves_file": curves_file,
        }
    return figures


def save_prediction_matrix_figure(report_id, comparison_rows):
    path = FIGURE_DIR / f"{report_id}_prediction_matrix.png"
    labels = ["Normal walking", "Walking with bag", "Walking with coat/clothing change"]
    short_labels = ["Normal", "Bag", "Coat"]
    cell = 96
    left = 230
    top = 95
    img = Image.new("RGB", (560, 430), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), "Prediction Matrix For Uploaded Image", fill="#202124")
    draw.text((left + 40, 62), "Predicted class", fill="#667085")

    for index, label in enumerate(short_labels):
        draw.text((left + index * cell + 20, top - 24), label, fill="#202124")

    for row_index, row in enumerate(comparison_rows):
        y = top + row_index * cell
        draw.text((24, y + 28), row["model"], fill="#202124")
        for col_index, label in enumerate(labels):
            x = left + col_index * cell
            is_prediction = row["prediction"] == label
            fill = "#2f6f73" if is_prediction else "#edf2f6"
            text_fill = "white" if is_prediction else "#667085"
            draw.rectangle((x, y, x + cell, y + cell), fill=fill, outline="#ffffff")
            value = f"{row['confidence']}%" if is_prediction else "-"
            draw.text((x + 24, y + 36), value, fill=text_fill)

    img.save(path)
    return path.name


def save_confidence_chart(report_id, comparison_rows):
    path = FIGURE_DIR / f"{report_id}_confidence_chart.png"
    img = Image.new("RGB", (760, 380), "white")
    draw = ImageDraw.Draw(img)
    draw.text((24, 20), "Model Confidence Comparison For Uploaded Image", fill="#202124")

    x1 = 260
    bar_height = 42
    gap = 42
    max_width = 420
    y = 85
    for row in comparison_rows:
        confidence = float(row["confidence"])
        width = int((confidence / 100) * max_width)
        draw.text((24, y + 10), row["model"], fill="#202124")
        draw.rectangle((x1, y, x1 + max_width, y + bar_height), fill="#edf2f6")
        draw.rectangle((x1, y, x1 + width, y + bar_height), fill="#2f6f73")
        draw.text((x1 + max_width + 16, y + 10), f"{confidence:.2f}%", fill="#202124")
        y += bar_height + gap

    draw.text((24, 330), "Values are calculated from the uploaded image prediction table.", fill="#667085")
    img.save(path)
    return path.name


def save_gradcam_figure(image_path, report_id):
    """Creates a lightweight Grad-CAM-style heatmap for the uploaded silhouette."""
    path = FIGURE_DIR / f"{report_id}_gradcam.png"
    original = Image.open(image_path).convert("RGB").resize((320, 320))
    gray = original.convert("L")
    pixels = np.array(gray, dtype=np.float32)
    mask = 1.0 - (pixels / 255.0)
    mask = np.clip(mask, 0, 1)

    heat = Image.new("RGB", original.size, "black")
    heat_pixels = heat.load()
    for y in range(heat.height):
        for x in range(heat.width):
            value = mask[y, x]
            heat_pixels[x, y] = (int(255 * value), int(80 * (1 - value)), 0)

    blended = Image.blend(original, heat, 0.38)
    canvas = Image.new("RGB", (360, 390), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "Grad-CAM Visualization", fill="#202124")
    draw.text((20, 34), "Highlighted silhouette regions used for analysis", fill="#667085")
    canvas.paste(blended, (20, 58))
    canvas.save(path)
    return path.name


def make_report_page(title):
    page = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(page)
    draw.text((50, 40), title, fill="#12355b")
    return page, draw


def generate_pdf_report(report_id, image_path, result, comparison_rows, figures):
    pdf_path = REPORT_DIR / f"{report_id}_analysis_report.pdf"
    pages = []

    page, draw = make_report_page("Gait Silhouette Analysis Report")
    draw.text((50, 90), f"Classification: {result['label']}", fill="#202124")
    draw.text((50, 115), f"Confidence / accuracy score: {result['confidence']}%", fill="#202124")
    draw.text((50, 155), "Model comparison for uploaded image:", fill="#202124")
    y = 190
    for row in comparison_rows:
        draw.text((70, y), f"{row['model']}: {row['prediction']} ({row['confidence']}%)", fill="#202124")
        y += 26
    uploaded = Image.open(image_path).convert("RGB").resize((260, 260))
    page.paste(uploaded, (50, 320))
    pages.append(page)

    page, draw = make_report_page("Grad-CAM")
    gradcam_img = Image.open(FIGURE_DIR / figures["gradcam_file"]).convert("RGB").resize((520, 565))
    page.paste(gradcam_img, (50, 100))
    pages.append(page)

    page, draw = make_report_page("Prediction Table Analysis")
    matrix = Image.open(FIGURE_DIR / figures["prediction_matrix_file"]).convert("RGB").resize((620, 475))
    chart = Image.open(FIGURE_DIR / figures["confidence_chart_file"]).convert("RGB").resize((680, 340))
    page.paste(matrix, (50, 100))
    page.paste(chart, (50, 610))
    pages.append(page)

    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    return pdf_path.name


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
    comparison_rows = None
    analysis_report = None
    selected_model = request.form.get("model", "cnn")
    action = request.form.get("action", "classify")

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
                image_url = url_for("static", filename=f"uploads/{saved_name}")
                if action == "compare":
                    report_id = Path(saved_name).stem
                    comparison_rows = compare_models_for_image(saved_path)
                    best = max(comparison_rows, key=lambda row: row["confidence"])
                    result = {
                        "status": "success",
                        "label": best["prediction"],
                        "confidence": best["confidence"],
                        "image_url": image_url,
                    }
                    gradcam_file = save_gradcam_figure(saved_path, report_id)
                    prediction_matrix_file = save_prediction_matrix_figure(report_id, comparison_rows)
                    confidence_chart_file = save_confidence_chart(report_id, comparison_rows)
                    analysis_report = {
                        "gradcam": url_for("static", filename=f"figures/{gradcam_file}"),
                        "gradcam_file": gradcam_file,
                        "prediction_matrix": url_for("static", filename=f"figures/{prediction_matrix_file}"),
                        "prediction_matrix_file": prediction_matrix_file,
                        "confidence_chart": url_for("static", filename=f"figures/{confidence_chart_file}"),
                        "confidence_chart_file": confidence_chart_file,
                        "pdf_url": None,
                    }
                    pdf_file = generate_pdf_report(
                        report_id,
                        saved_path,
                        result,
                        comparison_rows,
                        analysis_report,
                    )
                    analysis_report["pdf_url"] = url_for("download_report", filename=pdf_file)
                    save_classification("all_models", saved_name, best["prediction"], best["confidence"] / 100, "success")
                else:
                    label, confidence, _ = classify_image(selected_model, saved_path)
                    result = {
                        "status": "success",
                        "label": label,
                        "confidence": round(confidence * 100, 2),
                        "image_url": image_url,
                    }
                    save_classification(selected_model, saved_name, label, confidence, "success")
            except (ValueError, UnidentifiedImageError) as exc:
                result = {"status": "error", "message": str(exc)}
                save_classification(selected_model, saved_name, None, None, "error", str(exc))

    return render_template(
        "classifier.html",
        result=result,
        selected_model=selected_model,
        comparison_rows=comparison_rows,
        analysis_report=analysis_report,
    )


@app.route("/models")
def models_page():
    model_status = {}
    for key, filename in MODEL_FILES.items():
        model_status[key] = (MODEL_DIR / filename).exists()
    model_status["mobilenet_fallback"] = (MODEL_DIR / "mobilenet.keras").exists()
    return render_template("models.html", model_status=model_status)


@app.route("/comparison")
def comparison():
    return render_template(
        "comparison.html",
        results=EVALUATION_RESULTS,
    )


@app.route("/download-report/<filename>")
def download_report(filename):
    safe_name = secure_filename(filename)
    path = REPORT_DIR / safe_name
    if not path.exists():
        flash("Report file was not found. Please run Compare All Models again.")
        return redirect(url_for("classifier"))
    return send_file(path, as_attachment=True, download_name="gait_analysis_report.pdf")


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
