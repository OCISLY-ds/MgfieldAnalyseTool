import pandas as pd
import matplotlib.pyplot as plt

# Datei einlesen und gültige Zeilen sammeln
file_path = "/Users/florianvonbargen/Docker/testDocker/app/kp2412.tab"
valid_rows = []

with open(file_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) in [12, 13] and parts[0].isdigit() and len(parts[0]) == 6:
            if len(parts) == 12:
                parts.insert(10, "")  # fehlender Flag
            valid_rows.append(parts)

# In DataFrame umwandeln
columns = [
    "date", "kp_00_03", "kp_03_06", "kp_06_09", "kp_09_12",
    "kp_12_15", "kp_15_18", "kp_18_21", "kp_21_24",
    "kp_sum", "flag", "daily_sum", "kp_mean"
]
df = pd.DataFrame(valid_rows, columns=columns)

# Datum konvertieren
df["date"] = pd.to_datetime(df["date"], format="%y%m%d")

# KP-Werte umwandeln
def convert_kp(val):
    if isinstance(val, str):
        val = val.replace("o", ".0").replace("+", ".7").replace("-", ".3")
    try:
        return float(val)
    except:
        return None

kp_cols = [col for col in df.columns if col.startswith("kp_") and col != "kp_sum"]  # kp_sum ausschließen
df[kp_cols] = df[kp_cols].applymap(convert_kp)

# CSV-Datei speichern, um die Daten zu überprüfen
output_csv_path = "/Users/florianvonbargen/Docker/testDocker/app/kp_data.csv"
df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Die Daten wurden in die Datei {output_csv_path} geschrieben.")

# Daten für den Plot vorbereiten
plot_data = []
for col in kp_cols:
    times = pd.to_timedelta(kp_cols.index(col) * 3, unit="h")  # Zeitverschiebung für jede Spalte
    temp_df = df[["date", col]].copy()
    temp_df = temp_df.rename(columns={col: "value"})  # Spalte umbenennen
    temp_df["datetime"] = temp_df["date"] + times  # Zeitstempel hinzufügen
    plot_data.append(temp_df[["datetime", "value"]])

# Alle Daten in einem DataFrame zusammenführen
plot_df = pd.concat(plot_data).sort_values("datetime")

# Plot erzeugen
plt.figure(figsize=(12, 6))
plt.plot(plot_df["datetime"], plot_df["value"], marker='.', linestyle='-', color='royalblue', label="KP")
plt.axhline(5, color='red', linestyle='--', label='Sturm-Schwelle (KP >= 5)')
plt.title("KP-Index (3 Stunden)")
plt.xlabel("Datum und Zeit")
plt.ylabel("KP-Wert")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()