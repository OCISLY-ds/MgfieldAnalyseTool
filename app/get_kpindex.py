from ftplib import FTP
import os
from datetime import datetime

def download_kp_index_files(start_year=2023, ftp_host="ftp.gfz-potsdam.de", ftp_path="/pub/home/obs/kp-ap/tab/", local_dir="kp_index_files"):
    """
    Downloads KP Index files from the specified FTP server for all months since the start_year.
    """
    # Erstelle das lokale Verzeichnis, falls es nicht existiert
    os.makedirs(local_dir, exist_ok=True)

    # Verbinde mit dem FTP-Server
    ftp = FTP(ftp_host)
    ftp.login()  # Anonyme Anmeldung
    ftp.cwd(ftp_path)

    # Aktuelles Jahr und Monat
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # Iteriere über die Jahre und Monate seit dem Startjahr
    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            # Breche ab, wenn wir über das aktuelle Datum hinausgehen
            if year == current_year and month > current_month:
                break

            # Erstelle den Dateinamen basierend auf Jahr und Monat
            year_short = str(year)[-2:]  # Letzte zwei Ziffern des Jahres
            month_padded = f"{month:02d}"  # Monat mit führender Null
            filename = f"kp{year_short}{month_padded}.tab"

            # Lokaler Speicherpfad
            local_filepath = os.path.join(local_dir, filename)

            # Überspringe, wenn die Datei bereits heruntergeladen wurde
            if os.path.exists(local_filepath):
                print(f"Datei {filename} existiert bereits. Überspringe...")
                continue

            # Lade die Datei herunter
            try:
                with open(local_filepath, "wb") as file:
                    ftp.retrbinary(f"RETR {filename}", file.write)
                print(f"Datei {filename} erfolgreich heruntergeladen.")
            except Exception as e:
                print(f"Fehler beim Herunterladen der Datei {filename}: {e}")

    # Schließe die FTP-Verbindung
    ftp.quit()

if __name__ == "__main__":
    download_kp_index_files()