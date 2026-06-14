# Gait Silhouette Classifier

Simple Flask frontend and backend for classifying gait silhouette images.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

You are using Python 3.14, so the default `requirements.txt` runs the web app,
survey, admin panel, uploads, and the fallback silhouette classifier.

On Windows, install the local app dependencies with:

```bash
pip install -r requirements.txt
```

For deployment on Render or another Linux host, use `requirements-deploy.txt`
because it includes `gunicorn`.

For real TensorFlow `.keras` model prediction, use Python 3.11 or 3.12 and install:

```bash
pip install -r requirements-ml.txt
python app.py
```

Default admin login:

- Username: `admin`
- Password: `admin123`

Set `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and `SECRET_KEY` environment variables before deploying.

## Model files

Put trained model files in the `models` folder:

- `models/cnn.keras`
- `models/resnet.keras`
- `models/effnet.keras`

The notebook saved `mobilenet.keras`, not a ResNet model. If `models/resnet.keras` is missing but `models/mobilenet.keras` exists, the ResNet button uses MobileNet as a fallback.

If no model file exists, the app still runs and returns a basic silhouette heuristic result for demo purposes.

The model files are not created automatically by this Flask app. Run the notebook
training cells first, then copy the saved files into this project:

```text
F:\gait-classifier\models\cnn.keras
F:\gait-classifier\models\mobilenet.keras
F:\gait-classifier\models\effnet.keras
```

If you train a real ResNet model, save it as:

```text
F:\gait-classifier\models\resnet.keras
```

## Deploy

This project includes `render.yaml` for Render deployment. It uses Python 3.11
and installs `requirements-ml.txt` so TensorFlow can load the `.keras` files.

Free hosting is useful for demos, but it is not truly forever:

- Render free web services sleep after 15 minutes without traffic.
- Render free web services have an ephemeral filesystem, so uploaded files and
  the local SQLite database are not permanent.
- For permanent admin data, replace SQLite with a hosted database.

Before deploying, change the `ADMIN_PASSWORD` value in `render.yaml`.
