import os
import json
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO
from hapiclient import hapi
import numpy as np
import math
from tqdm import tqdm
import time
import csv
import plotly.graph_objects as go

app = Flask(__name__)
socketio = SocketIO(app)

# Ordner für gespeicherte Plots (zum Download)
DOWNLOAD_FOLDER = '/tmp/plots'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# Ordner für gespeicherte Daten
DATA_FOLDER = '/tmp/data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Benutzerdefinierter Filter für die Datumsformatierung
@app.template_filter('datetimeformat')
def datetimeformat(value):
    dt = datetime.strptime(value, '%Y-%m-%dT%H:%M')
    return dt.strftime('%d.%m.%Y %H:%M')

def load_valid_observatories(csv_file):
    valid_observatories = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Status'] == 'Open':
                valid_observatories[row['Code']] = {
                    'Name': row['Name'],
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude'])
                }
    print(f"Valid observatories loaded: {valid_observatories}")  # Debugging-Ausgabe
    return valid_observatories

def fetch_data_from_server(iaga_code, start, stop):
    server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    parameters = 'Field_Vector'
    opts = {'logging': True, 'usecache': True}
    dataset = f'{iaga_code.lower()}/best-avail/PT1M/xyzf'

    try:
        data, _ = hapi(server, dataset, parameters, start, stop, **opts)
        return data
    except Exception as e:
        print(f"Error fetching data for {iaga_code}: {str(e)}")
        return None

def load_or_fetch_data(iaga_code, start, stop):
    filename = os.path.join(DATA_FOLDER, f'{iaga_code}_{start}_{stop}.json')
    if os.path.exists(filename):
        print(f"Loading data from {filename}")
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
            # Konvertiere die Daten zurück in ein numpy-Array
            data = np.array([[item[0], np.array(item[1])] for item in data])
            return data
        except json.JSONDecodeError as e:
            print(f"Error loading JSON data from {filename}: {str(e)}")
            # Datei ist ungültig, löschen und neu abrufen
            os.remove(filename)
    
    print(f"Fetching data from server for {iaga_code}")
    data = fetch_data_from_server(iaga_code, start, stop)
    if data is not None:
        # Konvertiere die Daten in ein serialisierbares Format
        serializable_data = [[item[0].decode('utf-8'), item[1].tolist()] for item in data]
        with open(filename, 'w') as file:
            json.dump(serializable_data, file)
        # Konvertiere die Daten zurück in ein numpy-Array
        data = np.array([[item[0], np.array(item[1])] for item in serializable_data])
    return data

def process_data(iaga_codes, start, stop, valid_observatories, threshold):
    combined_data = {}
    filtered_data_info = []
    start_time = time.time()
    total_filtered = 0
    for i, iaga_code in enumerate(tqdm(iaga_codes, desc="Processing data")):
        print(f"Processing station: {iaga_code}")  # Debugging-Ausgabe
        if iaga_code == 'BUE':
            observatory_name = 'BUE'
        else:
            observatory_name = valid_observatories[iaga_code]['Name']

        data = load_or_fetch_data(iaga_code, start, stop)

        if isinstance(data, np.ndarray) and data.ndim == 1:
            timestamps = [item[0].decode('utf-8') for item in data]
            vectors = np.array([item[1] for item in data])

            magnitudes = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in vectors]
            filtered_data = [(t, m) for t, m in zip(timestamps, magnitudes) if m <= threshold]

            total_filtered += len(magnitudes) - len(filtered_data)

            for t, m in zip(timestamps, magnitudes):
                if m > threshold:
                    readable_time = None
                    for fmt in ('%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%MZ', '%Y-%m-%dT%H:%M:%SZ'):
                        try:
                            readable_time = datetime.strptime(t, fmt).strftime('%d.%m.%Y %H:%M:%S')
                            break
                        except ValueError:
                            continue
                    if readable_time is None:
                        readable_time = t  # Fallback, falls kein Format passt
                    filtered_data_info.append(f"{readable_time} - Station {iaga_code}: {m}")

            if filtered_data:
                filtered_timestamps, filtered_magnitudes = zip(*filtered_data)
                combined_data[iaga_code] = (filtered_timestamps, filtered_magnitudes, observatory_name)
            else:
                print(f"Error: Alle Datenpunkte für Station {iaga_code} überschreiten den Schwellenwert von {threshold}.")
        else:
            print(f"Error: Unerwartetes Datenformat für Station {iaga_code}.")

        # Senden Sie den Fortschritt und die geschätzte Restzeit an den Client
        elapsed_time = time.time() - start_time
        progress = int((i + 1) / len(iaga_codes) * 100)
        estimated_total_time = elapsed_time / (i + 1) * len(iaga_codes)
        remaining_time = estimated_total_time - elapsed_time
        socketio.emit('progress', {'progress': progress, 'remaining_time': int(remaining_time)})

    if combined_data:
        plot_filename, plot_changes_filename = save_and_plot_magnitude(combined_data, start, stop, valid_observatories)
        save_combined_data_to_csv(combined_data, start, stop)
        correlation_matrix = calculate_correlations(combined_data)
    else:
        correlation_matrix = None
    
    print(f"Anzahl der gefilterten Datenpunkte: {total_filtered}")
    return combined_data, total_filtered, filtered_data_info, correlation_matrix

def calculate_correlations(combined_data):
    print("Calculating correlations...")  # Debugging-Ausgabe
    stations = list(combined_data.keys())
    correlation_matrix = pd.DataFrame(index=stations, columns=stations)

    # Alle Zeitstempel sammeln und sortieren
    all_timestamps = sorted(set(ts for timestamps, _, _ in combined_data.values() for ts in timestamps))

    # Interpolierte Magnituden für alle Stationen
    interpolated_data = {}
    for station in stations:
        timestamps, magnitudes, _ = combined_data[station]
        magnitudes_series = pd.Series(magnitudes, index=pd.to_datetime(timestamps))
        magnitudes_series = magnitudes_series.reindex(pd.to_datetime(all_timestamps)).interpolate()
        interpolated_data[station] = magnitudes_series

    for i, station1 in enumerate(stations):
        for j, station2 in enumerate(stations):
            if i <= j:  # Nur obere Dreiecksmatrix berechnen
                magnitudes1 = interpolated_data[station1]
                magnitudes2 = interpolated_data[station2]
                correlation = np.corrcoef(magnitudes1, magnitudes2)[0, 1]
                correlation_matrix.loc[station1, station2] = correlation
                correlation_matrix.loc[station2, station1] = correlation

    print("Correlations calculated.")  # Debugging-Ausgabe
    return correlation_matrix

def save_combined_data_to_csv(combined_data, start, stop):
    print("Saving combined data to CSV...")  # Debugging-Ausgabe
    output_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'combined')
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'combined_data_{start[:10]}_to_{stop[:10]}.csv')

    # Alle Zeitstempel sammeln
    all_timestamps = sorted(set(ts for timestamps, _, _ in combined_data.values() for ts in timestamps))

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Überschriftenzeile schreiben
        header = ['Timestamp'] + [f'{observatory_name} ({iaga_code})' for iaga_code, (_, _, observatory_name) in combined_data.items()]
        writer.writerow(header)

        # Datenzeilen schreiben
        for ts in all_timestamps:
            row = [ts]
            for iaga_code in combined_data.keys():
                if ts in combined_data[iaga_code][0]:
                    index = combined_data[iaga_code][0].index(ts)
                    row.append(combined_data[iaga_code][1][index])
                else:
                    row.append('')
            writer.writerow(row)

    print(f'Kombinierte Daten als CSV gespeichert: {csv_filename}')

def save_and_plot_magnitude(combined_data, start, stop, valid_observatories):
    print("Saving and plotting magnitude...")  # Debugging-Ausgabe
    output_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'combined')
    os.makedirs(output_dir, exist_ok=True)

    # Plot der Magnetfeldstärke
    fig = go.Figure()
    for iaga_code, (timestamps, magnitudes, observatory_name) in combined_data.items():
        fig.add_trace(go.Scatter(x=timestamps, y=magnitudes, mode='lines', name=f'{observatory_name} ({iaga_code})'))

    fig.update_layout(
        title='Vergleich der Magnetfeldbeträge zwischen den Observatorien',
        xaxis_title='Zeit',
        yaxis_title='Magnetfeldstärke (nT)',
        xaxis=dict(tickangle=45),
        legend_title='Observatorien',
        template='plotly_white',  # Setzen des Plotly White Themes
        margin=dict(l=0, r=0, t=0, b=0)  # Entfernen der Ränder
    )

    plot_filename = f'combined_magnitude_{start[:10]}_to_{stop[:10]}.html'
    plot_filepath = os.path.join(output_dir, plot_filename)
    fig.write_html(plot_filepath)

    print(f'Kombinierter Graph gespeichert: {plot_filepath}')

    # Plot der Veränderungen der Magnetfeldstärke
    fig_changes = go.Figure()
    for iaga_code, (timestamps, magnitudes, observatory_name) in combined_data.items():
        # Verwenden Sie den ersten Wert als Basislinie
        baseline = magnitudes[0]
        changes = [m - baseline for m in magnitudes]
        fig_changes.add_trace(go.Scatter(x=timestamps, y=changes, mode='lines', name=f'{observatory_name} ({iaga_code})'))

    fig_changes.update_layout(
        title='Veränderungen der Magnetfeldstärke zwischen den Observatorien',
        xaxis_title='Zeit',
        yaxis_title='Veränderung der Magnetfeldstärke (nT)',
        xaxis=dict(tickangle=45),
        legend_title='Observatorien',
        template='plotly_white',  # Setzen des Plotly White Themes
        margin=dict(l=0, r=0, t=0, b=0)  # Entfernen der Ränder
    )

    plot_changes_filename = f'changes_magnitude_{start[:10]}_to_{stop[:10]}.html'
    plot_changes_filepath = os.path.join(output_dir, plot_changes_filename)
    fig_changes.write_html(plot_changes_filepath)

    print(f'Graph der Veränderungen gespeichert: {plot_changes_filepath}')

    # Leaflet-Karte erstellen
    map_data = []
    for iaga_code, (timestamps, magnitudes, observatory_name) in combined_data.items():
        observatory = valid_observatories[iaga_code]
        map_data.append({
            'iaga_code': iaga_code,
            'observatory_name': observatory_name,
            'latitude': observatory['Latitude'],
            'longitude': observatory['Longitude']
        })

    map_data.append({
        'iaga_code': 'BUE',
        'observatory_name': 'Bützflethermoor',
        'latitude': 53.651,
        'longitude': 9.424
    })

    map_data_filename = os.path.join(output_dir, 'map_data.json')
    with open(map_data_filename, 'w') as f:
        json.dump(map_data, f)

    print(f'Karten-Daten gespeichert: {map_data_filename}')

    return plot_filename, plot_changes_filename

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        stations_input = request.form.get('station')
        distance = request.form.get('distance', type=float)
        start = request.form.get('start', '2024-09-01T00:00')
        stop = request.form.get('stop', '2024-10-01T00:00')
        threshold = request.form.get('threshold', type=float, default=100000)
        
        stations = []
        if stations_input:
            stations = [station.strip().upper() for station in stations_input.split(',') if len(station.strip()) == 3]

        print(f"Benutzer hat folgende Zeiten eingegeben: Start = {start}, End = {stop}, Stationen = {stations}, Abstand = {distance}, Schwellenwert = {threshold}")

        csv_file = os.path.join(os.getcwd(), 'intermagnet/IAGAlist.csv')
        valid_observatories = load_valid_observatories(csv_file)

        if distance is not None:
            bue_longitude = 9.424
            bue_latitude = 53.651
            for code, observatory in valid_observatories.items():
                if abs(observatory['Latitude'] - bue_latitude) <= distance:
                    stations.append(code)

        stations = list(set(stations))  # Duplikate entfernen

        invalid_stations = [station for station in stations if station not in valid_observatories]
        if invalid_stations:
            print(f"Fehler: Stationen {invalid_stations} nicht in valid_observatories")  # Debugging-Ausgabe
            return render_template('index.html', message=f"Fehler: Ungültige Stationen {invalid_stations}.", start=start, stop=stop)

        # Fügen Sie die manuell hinzugefügte Station BUE hinzu
        if 'BUE' not in stations:
            stations.append('BUE')

        combined_data, total_filtered, filtered_data_info, correlation_matrix = process_data(stations, start, stop, valid_observatories, threshold)

        if not combined_data:
            return render_template('index.html', message="Fehler: Keine Daten gefunden.", start=start, stop=stop, stations=stations)

        plot_filename, plot_changes_filename = save_and_plot_magnitude(combined_data, start, stop, valid_observatories)
        map_plot_filename = f'stations_map_{start[:10]}_to_{stop[:10]}.html'
        csv_filename = f'combined_data_{start[:10]}_to_{stop[:10]}.csv'

        return render_template('index.html', message=f"Plots erfolgreich erstellt! Anzahl der gefilterten Datenpunkte: {total_filtered}", plot_url=f'combined/{plot_filename}', plot_changes_url=f'combined/{plot_changes_filename}', map_plot_url=f'combined/{map_plot_filename}', csv_url=f'combined/{csv_filename}', start=start, stop=stop, stations=stations, filtered_data_info=filtered_data_info, correlation_matrix=correlation_matrix)
    
    return render_template('index.html', message="Bitte die Start- und Endzeit eingeben.", start=None, stop=None)    
@app.route('/download/<path:filename>')
def download_file(filename):
    print(f"Download requested for file: {filename}")  # Debugging-Ausgabe
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)