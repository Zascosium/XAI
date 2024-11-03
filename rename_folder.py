import os
import pandas as pd

# Pfad zur CSV-Datei und zum Ordner mit den Unterordnern
csv_path = 'GTSRB/GTSRB_Class_Names.csv'  # Pfad zur CSV-Datei
root_folder = 'GTSRB/Final_Training/Images'  # Root-Ordner, der die Ordner 00000 bis 00042 enthält

# CSV-Datei laden und die erste Zeile überspringen
try:
    df = pd.read_csv(csv_path, sep=';', engine='python', skiprows=1)  # Erste Zeile überspringen
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    exit(1)

# Überprüfen, ob die CSV-Datei die erwarteten zwei Spalten enthält
if df.shape[1] < 2:
    print("Die CSV-Datei enthält weniger als zwei Spalten. Bitte überprüfen Sie die Datei.")
    exit(1)

# Spaltennamen setzen
folder_column = df.columns[0]  # Erste Spalte: Ordnernummer (z.B., 00000, 00001, ...)
class_name_column = df.columns[1]  # Zweite Spalte: Klassenname (z.B., 20_kmh, etc.)

# Durch alle Zeilen der CSV-Datei gehen
for index, row in df.iterrows():
    try:
        # Ordnernummer formatieren
        folder_number = f"{int(row[folder_column]):05d}"  # Ordnernummer mit führenden Nullen, z.B., 00000
        class_name = row[class_name_column]  # Klassenname, z.B., 20_kmh

        # Vollständiger Pfad des aktuellen und neuen Ordners
        current_folder_path = os.path.join(root_folder, folder_number)
        new_folder_path = os.path.join(root_folder, class_name)

        # Ordner umbenennen, wenn der aktuelle Ordner existiert
        if os.path.exists(current_folder_path):
            os.rename(current_folder_path, new_folder_path)
            print(f"Ordner '{current_folder_path}' wurde in '{new_folder_path}' umbenannt.")
        else:
            print(f"Ordner '{current_folder_path}' existiert nicht und konnte nicht umbenannt werden.")
    except ValueError as ve:
        print(f"Fehler bei Zeile {index}: {ve}")
