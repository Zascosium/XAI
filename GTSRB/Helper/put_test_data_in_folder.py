import os
import pandas as pd
import shutil

# path to CSV-file and the image folder
csv_path = '../GTSRB/Final_Test/Images/GT-final_test.csv'  
image_folder = "../GTSRB/Final_Test/Images" 
target_folder = '../GTSRB/Final_Test/Images' 

# create target folder if it does not exist
os.makedirs(target_folder, exist_ok=True)

# load CSV-file
try:
    df = pd.read_csv(csv_path, sep=';', engine='python') 
except Exception as e:
    print(f"Fehler beim Laden der CSV-Datei: {e}")
    exit(1)

if 'Filename' not in df.columns or 'ClassId' not in df.columns:
    print("Die CSV-Datei enthält nicht die benötigten Spalten 'Filename' und 'ClassId'.")
    exit(1)

for index, row in df.iterrows():
    filename = row['Filename']        
    class_id = int(row['ClassId'])    
    class_folder = f"{class_id:05d}"   
    
    # path to class folder
    target_class_folder = os.path.join(target_folder, class_folder)
    os.makedirs(target_class_folder, exist_ok=True)  
    
    # path to source and target image files
    source_image_path = os.path.join(image_folder, filename)
    target_image_path = os.path.join(target_class_folder, filename)
    
    # move the source image to the target
    if os.path.exists(source_image_path):
        shutil.move(source_image_path, target_image_path)
        print(f"Bild '{filename}' wurde in '{target_class_folder}' verschoben.")
    else:
        print(f"Bild '{filename}' nicht gefunden und konnte nicht verschoben werden.")
