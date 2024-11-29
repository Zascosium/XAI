import os
import pandas as pd
import shutil

# Pfade zur CSV-Datei und zum Bildordner
csv_path = 'GTSRB/Final_Test/Images/GT-final_test.csv'  # Pfad zur neuen CSV-Datei
image_folder = "C:/Users/v814u63\Documents/Uni/5. Semester/XAI/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"  # Verzeichnis, in dem die Bilder und die CSV-Datei liegen
target_folder = 'test_images'  # Übergeordneter Ordner, in dem die Klassenordner erstellt werden

# Zielordner erstellen, falls nicht vorhanden
os.makedirs(target_folder, exist_ok=True)

# CSV-Datei laden
try:
    df = pd.read_csv(csv_path, sep=';', engine='python')  # CSV mit Semikolon als Trennzeichen
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    exit(1)

# Überprüfen, ob die CSV-Datei die benötigten Spalten enthält
if 'Filename' not in df.columns or 'ClassId' not in df.columns:
    print("Die CSV-Datei enthält nicht die benötigten Spalten 'Filename' und 'ClassId'.")
    exit(1)

# Bilder in die entsprechenden Klassenordner verschieben
for index, row in df.iterrows():
    filename = row['Filename']         # Bilddateiname
    class_id = int(row['ClassId'])     # Klassennummer
    class_folder = f"{class_id:05d}"   # Klassenordnername (z.B., 00000, 00001, ..., 00042)
    
    # Pfad zum Zielordner für die aktuelle Klasse
    target_class_folder = os.path.join(target_folder, class_folder)
    os.makedirs(target_class_folder, exist_ok=True)  # Klassenordner erstellen, falls nicht vorhanden
    
    # Pfade zum Quellbild und Zielpfad
    source_image_path = os.path.join(image_folder, filename)
    target_image_path = os.path.join(target_class_folder, filename)
    
    # Bild verschieben, falls es existiert
    if os.path.exists(source_image_path):
        shutil.move(source_image_path, target_image_path)
        print(f"Bild '{filename}' wurde in '{target_class_folder}' verschoben.")
    else:
        print(f"Bild '{filename}' nicht gefunden und konnte nicht verschoben werden.")
