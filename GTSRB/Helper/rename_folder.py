import os
import pandas as pd

# Path to CSV-file and the root folder
csv_path = '../GTSRB/GTSRB_Class_Names.csv'  
root_folder = '../GTSRB/Final_Test/Images'  

# load CSV-file and skip first row
try:
    df = pd.read_csv(csv_path, sep=';', engine='python', skiprows=1)  
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    exit(1)

# check format of the CSV-file
if df.shape[1] < 2:
    print("Die CSV-Datei enthält weniger als zwei Spalten. Bitte überprüfen Sie die Datei.")
    exit(1)

folder_column = df.columns[0]  
class_name_column = df.columns[1]  

# iterate over all rows and rename the folders
for index, row in df.iterrows():
    try:
        folder_number = f"{int(row[folder_column]):05d}" 
        class_name = row[class_name_column]  

        current_folder_path = os.path.join(root_folder, folder_number)
        new_folder_path = os.path.join(root_folder, class_name)

        if os.path.exists(current_folder_path):
            os.rename(current_folder_path, new_folder_path)
            print(f"Ordner '{current_folder_path}' wurde in '{new_folder_path}' umbenannt.")
        else:
            print(f"Ordner '{current_folder_path}' existiert nicht und konnte nicht umbenannt werden.")
    except ValueError as ve:
        print(f"Fehler bei Zeile {index}: {ve}")
