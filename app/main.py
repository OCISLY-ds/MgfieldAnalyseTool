from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import safe_join
import plotly.graph_objects as go
import os
import pandas as pd
from hapiclient import hapi
from datetime import datetime, timedelta, timezone
import csv
import numpy as np
import math
from tqdm import tqdm
import concurrent.futures
import io
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Ordner für gespeicherte Plots (zum Download)
DOWNLOAD_FOLDER = '/tmp/plots'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

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

def fetch_data(iaga_code, server, parameters, start, stop, opts):
    dataset = f'{iaga_code.lower()}/best-avail/PT1M/xyzf'
    try:
        data, _ = hapi(server, dataset, parameters, start, stop, **opts)
        return iaga_code, data
    except Exception as e:
        print(f"Error fetching data for {iaga_code}: {str(e)}")
        return iaga_code, None

def process_data(iaga_codes, start, stop, valid_observatories, filter_values):
    server = 'https://imag-data.bgs.ac.uk/GIN_V1/hapi'
    parameters = 'Field_Vector'
    opts = {'logging': True, 'usecache': True}

    combined_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_data, iaga_code, server, parameters, start, stop, opts): iaga_code for iaga_code in iaga_codes}
        for future in concurrent.futures.as_completed(futures):
            iaga_code = futures[future]
            try:
                iaga_code, data = future.result()
                if data is not None:
                    observatory_name = valid_observatories[iaga_code]['Name']
                    if isinstance(data, np.ndarray) and data.ndim == 1:
                        timestamps = [item[0].decode('utf-8') for item in data]
                        vectors = np.array([item[1] for item in data])
                        magnitudes = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in vectors]
                        if filter_values:
                            filtered_data = [(t, m) for t, m in zip(timestamps, magnitudes) if m <= 100000]
                        else:
                            filtered_data = [(t, m) for t, m in zip(timestamps, magnitudes)]
                        if filtered_data:
                            filtered_timestamps, filtered_magnitudes = zip(*filtered_data)
                            combined_data[iaga_code] = (filtered_timestamps, filtered_magnitudes, observatory_name)
                        else:
                            print(f"Error: Alle Datenpunkte für Station {iaga_code} überschreiten den Schwellenwert von 100.000.")
                    else:
                        print(f"Error: Unerwartetes Datenformat für Station {iaga_code}.")
            except Exception as e:
                print(f"Error processing data for {iaga_code}: {str(e)}")

    if combined_data:
        print("Daten erfolgreich kombiniert. Starte Plot- und CSV-Erstellung...")
        save_and_plot_magnitude(combined_data, start, stop, valid_observatories)
        save_combined_data_to_csv(combined_data, start, stop)
        print("Plot- und CSV-Erstellung abgeschlossen.")
    
    return combined_data

def save_combined_data_to_csv(combined_data, start, stop):
    output_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'combined')
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'combined_data_{start[:10]}_to_{stop[:10]}.csv')

    # Alle Zeitstempel sammeln und sortieren
    all_timestamps = sorted(set(ts for timestamps, _, _ in combined_data.values() for ts in timestamps))

    # Erstellen eines DataFrames für jede Station
    data_frames = []
    for iaga_code, (timestamps, magnitudes, observatory_name) in combined_data.items():
        df = pd.DataFrame({'Timestamp': timestamps, f'{observatory_name} ({iaga_code})': magnitudes})
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        data_frames.append(df)

    # Zusammenführen aller DataFrames basierend auf den Zeitstempeln
    combined_df = pd.concat(data_frames, axis=1).sort_index()

    # Speichern des kombinierten DataFrames als CSV in einem In-Memory-Objekt
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=True, encoding='utf-8')
    csv_content = csv_buffer.getvalue()

    # Schreiben des CSV-Inhalts in die Datei
    with open(csv_filename, 'w', encoding='utf-8') as file:
        file.write(csv_content)

    print(f'Kombinierte Daten als CSV gespeichert: {csv_filename}')

def save_and_plot_magnitude(combined_data, start, stop, valid_observatories):
    output_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'combined')
    os.makedirs(output_dir, exist_ok=True)

    print("Starte Erstellung des Magnetfeldstärke-Plots...")  # Debugging-Ausgabe
    # Plot der Magnetfeldstärke
    fig = go.Figure()
    for iaga_code, (timestamps, magnitudes, observatory_name) in combined_data.items():
        # Verwenden Sie den ersten Wert als Basislinie
        baseline = magnitudes[0]
        adjusted_magnitudes = [m - baseline for m in magnitudes]
        fig.add_trace(go.Scatter(x=timestamps, y=adjusted_magnitudes, mode='lines', name=f'{observatory_name} ({iaga_code})'))

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

    plot_jpg_filename = f'combined_magnitude_{start[:10]}_to_{stop[:10]}.jpg'
    plot_jpg_filepath = os.path.join(output_dir, plot_jpg_filename)
    fig.write_image(plot_jpg_filepath, format='jpg')

    print(f'Kombinierter Graph gespeichert: {plot_filepath}')
    print(f'Kombinierter Graph als JPG gespeichert: {plot_jpg_filepath}')

    print("Starte Erstellung der Weltkarte...")  # Debugging-Ausgabe
    # Plot der Weltkarte mit den Stationen
    map_fig = go.Figure()

    for iaga_code in combined_data.keys():
        observatory = valid_observatories[iaga_code]
        map_fig.add_trace(go.Scattergeo(
            lon=[observatory['Longitude']],
            lat=[observatory['Latitude']],
            text=iaga_code,
            mode='markers+text',
            textposition='top center',  # Text über dem Marker
            marker=dict(
                size=8,
                color='blue',
                symbol='circle'
            ),
            name=f"{observatory['Name']} ({iaga_code})"
        ))

    map_fig.update_layout(
        title='Position der ausgewählten Stationen',
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            subunitcolor='rgb(217, 217, 217)',
            countrycolor='rgb(217, 217, 217)',
            showcountries=True,
            showcoastlines=True,
            coastlinecolor='rgb(217, 217, 217)',
            projection_type='equirectangular',
            lonaxis=dict(
                range=[-20, 40]  # Längengradbereich anpassen
            ),
            lataxis=dict(
                range=[30, 70]  # Breitengradbereich anpassen
            )
        ),
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0)  # Entfernen der Ränder
    )

    map_plot_filename = f'stations_map_{start[:10]}_to_{stop[:10]}.html'
    map_plot_filepath = os.path.join(output_dir, map_plot_filename)
    map_fig.write_html(map_plot_filepath)

    map_plot_jpg_filename = f'stations_map_{start[:10]}_to_{stop[:10]}.jpg'
    map_plot_jpg_filepath = os.path.join(output_dir, map_plot_jpg_filename)
    map_fig.write_image(map_plot_jpg_filepath, format='jpg')

    print(f'Weltkarte der Stationen gespeichert: {map_plot_filepath}')
    print(f'Weltkarte der Stationen als JPG gespeichert: {map_plot_jpg_filepath}')
    print("Erstellung der Weltkarte abgeschlossen.")  # Debugging-Ausgabe

    print("Starte Erstellung der Korrelationsmatrix...")  # Debugging-Ausgabe
    # Berechnung der Korrelationsmatrix
    bue_latitude = 53.651
    stations = sorted(combined_data.keys(), key=lambda code: abs(valid_observatories[code]['Latitude'] - bue_latitude))
    correlation_matrix = pd.DataFrame(index=stations, columns=stations)

    # Alle Zeitstempel sammeln und sortieren
    all_timestamps = sorted(set(ts for timestamps, _, _ in combined_data.values() for ts in timestamps))

    # Erstellen eines DataFrames für die interpolierten Magnituden
    interpolated_df = pd.DataFrame(index=pd.to_datetime(all_timestamps))
    for station in stations:
        timestamps, magnitudes, _ = combined_data[station]
        magnitudes_series = pd.Series(magnitudes, index=pd.to_datetime(timestamps))
        interpolated_df[station] = magnitudes_series

    # Interpolieren der fehlenden Werte
    interpolated_df = interpolated_df.interpolate()

    for i, station1 in enumerate(stations):
        for j, station2 in enumerate(stations):
            if i <= j:  # Nur obere Dreiecksmatrix berechnen
                magnitudes1 = interpolated_df[station1]
                magnitudes2 = interpolated_df[station2]
                correlation = np.corrcoef(magnitudes1, magnitudes2)[0, 1]
                correlation_matrix.loc[station1, station2] = correlation
                correlation_matrix.loc[station2, station1] = correlation

    # Erstellen des Textes für die Korrelationswerte
    text = correlation_matrix.applymap(lambda x: f'{x:.2f}')

    # Berechnung der Abstände in Kilometern auf der Breitenachse zur BUE-Station
    def calculate_distance(lat1, lat2):
        # Radius der Erde in Kilometern
        R = 6371.0
        # Berechnung der Entfernung in Kilometern
        return abs(lat1 - lat2) * (math.pi / 180) * R

    station_labels_x = [f"{valid_observatories[code]['Name']} ({code})<br>{calculate_distance(bue_latitude, valid_observatories[code]['Latitude']):.0f} km" for code in stations]
    station_labels_y = [f"{code}" for code in stations]

    # Plot der Korrelationsmatrix
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values.astype(float),
        x=station_labels_x,
        y=station_labels_y,
        colorscale='Greens',
        zmin=0,
        zmax=1,
        text=text.values,
        texttemplate="%{text}",
        colorbar=dict(title='Korrelation')
    ))

    fig_corr.update_layout(
        title='Korrelationsmatrix der Magnetfeldstärken',
        xaxis_title='Station',
        yaxis_title='Station',
        template='plotly_white',
        margin=dict(l=0, r=0, t=0, b=0)  # Entfernen der Ränder
    )

    corr_plot_filename = f'correlation_matrix_{start[:10]}_to_{stop[:10]}.html'
    corr_plot_filepath = os.path.join(output_dir, corr_plot_filename)
    fig_corr.write_html(corr_plot_filepath)

    corr_plot_jpg_filename = f'correlation_matrix_{start[:10]}_to_{stop[:10]}.jpg'
    corr_plot_jpg_filepath = os.path.join(output_dir, corr_plot_jpg_filename)
    fig_corr.write_image(corr_plot_jpg_filepath, format='jpg')

    print(f'Korrelationsmatrix gespeichert: {corr_plot_filepath}')
    print(f'Korrelationsmatrix als JPG gespeichert: {corr_plot_jpg_filepath}')
    print("Erstellung der Korrelationsmatrix abgeschlossen.")  # Debugging-Ausgabe

    return plot_filename, map_plot_filename, corr_plot_filename, plot_jpg_filename, map_plot_jpg_filename, corr_plot_jpg_filename

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start_time = time.time()  # Startzeit messen

        selection_method = request.form.get('selection_method')
        start = request.form.get('start', '2024-09-01T00:00')
        stop = request.form.get('stop', '2024-10-01T00:00')

        # Konvertiere das Datum in das benötigte Format
        start = datetime.strptime(start, '%Y-%m-%dT%H:%M').strftime('%Y-%m-%dT%H:%M')
        stop = datetime.strptime(stop, '%Y-%m-%dT%H:%M').strftime('%Y-%m-%dT%H:%M')

        csv_file = os.path.join(os.getcwd(), 'intermagnet/IAGAlist.csv')
        valid_observatories = load_valid_observatories(csv_file)

        stations = []

        if selection_method == 'latitude':
            distance = request.form.get('distance', type=float)
            exclude_stations_input = request.form.get('exclude_stations')
            exclude_stations = [station.strip().upper() for station in exclude_stations_input.split(',')] if exclude_stations_input else []
            print(f"Benutzer hat folgende Zeiten eingegeben: Start = {start}, End = {stop}, Abstand = {distance}, Ausgeschlossene Stationen = {exclude_stations}")

            if distance is not None:
                bue_latitude = 53.651
                for code, observatory in valid_observatories.items():
                    if abs(observatory['Latitude'] - bue_latitude) <= distance and code not in exclude_stations:
                        stations.append(code)

        elif selection_method == 'station_codes':
            stations_input = request.form.get('station')
            if stations_input:
                stations = [station.strip().upper() for station in stations_input.split(',') if len(station.strip()) == 3]
            print(f"Benutzer hat folgende Zeiten eingegeben: Start = {start}, End = {stop}, Stationen = {stations}")

        stations = list(set(stations))  # Duplikate entfernen

        invalid_stations = [station for station in stations if station not in valid_observatories]
        if invalid_stations:
            print(f"Fehler: Stationen {invalid_stations} nicht in valid_observatories")  # Debugging-Ausgabe
            return render_template('index.html', message=f"Fehler: Ungültige Stationen {invalid_stations}.", start=start, stop=stop)

        # Fügen Sie die manuell hinzugefügte Station BUE hinzu
        if 'BUE' not in stations:
            stations.append('BUE')
            valid_observatories['BUE'] = {'Name': 'BUE', 'Latitude': 53.651, 'Longitude': 9.424}

        filter_values = request.form.get('filter_values', 'true').lower() == 'true'
        combined_data = process_data(stations, start, stop, valid_observatories, filter_values)

        if combined_data:
            plot_filename, map_plot_filename, corr_plot_filename, plot_jpg_filename, map_plot_jpg_filename, corr_plot_jpg_filename = save_and_plot_magnitude(combined_data, start, stop, valid_observatories)
            csv_filename = f'combined_data_{start[:10]}_to_{stop[:10]}.csv'
            end_time = time.time()  # Endzeit messen
            elapsed_time = end_time - start_time  # Zeitdifferenz berechnen
            station_data = [{'name': valid_observatories[code]['Name'], 'latitude': valid_observatories[code]['Latitude'], 'longitude': valid_observatories[code]['Longitude']} for code in combined_data.keys()]
            return render_template('index.html', message="Plots erfolgreich erstellt!", plot_url=f'combined/{plot_filename}', map_plot_url=f'combined/{map_plot_filename}', corr_plot_url=f'combined/{corr_plot_filename}', plot_jpg_url=f'combined/{plot_jpg_filename}', map_plot_jpg_url=f'combined/{map_plot_jpg_filename}', corr_plot_jpg_url=f'combined/{corr_plot_jpg_filename}', csv_url=f'combined/{csv_filename}', start=start, stop=stop, elapsed_time=elapsed_time, station_data=station_data)
        else:
            return render_template('index.html', message="Fehler: Keine Daten gefunden.", start=start, stop=stop)
    
    return render_template('index.html', message="Bitte die Start- und Endzeit eingeben.", start=None, stop=None)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

    #xxxx