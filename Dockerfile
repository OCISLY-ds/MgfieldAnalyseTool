# Basis-Image mit Python
FROM python:3.11

# Arbeitsverzeichnis setzen
WORKDIR /app

# Abhängigkeiten kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code kopieren
COPY app /app

# Flask-App starten
CMD ["python", "main.py"]