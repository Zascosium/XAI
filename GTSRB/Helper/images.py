# function dump at the beginning of the project

import os
import pandas as pd
from PIL import Image

def count_all_images(folder_path):
    total_images_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # check if the file is an image
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                total_images_count += 1

    print(f"Gesamtanzahl der Bilder: {total_images_count}")
    return total_images_count



def count_small_images(folder_path, min_width, min_height):
    small_images_count = 0
    total_images_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        total_images_count += 1                         
                        if width < min_width or height < min_height:
                            small_images_count += 1
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    
    print(f"\nGesamtanzahl der Bilder: {total_images_count}")
    print(f"Anzahl der Bilder kleiner als {min_width}x{min_height}: {small_images_count}")



def calculate_largest_aspect_ratio_difference(folder_path):
    largest_difference = 0
    image_with_largest_difference = ""

    # iterate over all subfolders of the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # calculate the difference in percent from a squared image
                        if width != height:
                            difference = abs(width - height) / max(width, height) * 100
                            
                            if difference > largest_difference:
                                largest_difference = difference
                                image_with_largest_difference = file_path
                                aspect_ratio_of_largest_difference = f"{width}:{height}"
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    
    print(f"Größter Seitenverhältnis-Unterschied: {largest_difference:.2f}%")
    print(f"Width x Height:  {aspect_ratio_of_largest_difference}")
    return largest_difference, image_with_largest_difference



def calculate_percentage_of_images_with_large_aspect_ratio_difference(folder_path, threshold_percentage=15):
    total_images_count = 0
    images_with_large_difference_count = 0

    # iterate over all subfolders of the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    # open the image and check the resolution
                    with Image.open(file_path) as img:
                        width, height = img.size
                        total_images_count += 1 
                        
                        if width != height:
                            difference = abs(width - height) / max(width, height) * 100
                            
                            if difference > threshold_percentage:
                                images_with_large_difference_count += 1
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    
    if total_images_count > 0:
        percentage = (images_with_large_difference_count / total_images_count) * 100
    else:
        percentage = 0
    
    print(f"Prozentanzahl der Bilder mit einer Abweichung größer als {threshold_percentage}%: {percentage:.2f}%")
    print(f"Gesamtanzahl der Bilder: {total_images_count}")
    print(f"Anzahl der Bilder mit einer Abweichung größer als {threshold_percentage}%: {images_with_large_difference_count}")
    
    return percentage, images_with_large_difference_count, total_images_count



def count_images_falling_below_threshold_or_large_aspect_ratio_difference(folder_path, size_threshold=35, ratio_threshold=10):
    total_images_count = 0
    images_falling_below_or_large_ratio_count = 0

    # iterate over all subfolders of the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        total_images_count += 1  
                        
                        if width < size_threshold or height < size_threshold:
                            images_falling_below_or_large_ratio_count += 1
                        else:
                            difference = abs(width - height) / max(width, height) * 100
                            
                            # check if the difference is greater than the specified threshold
                            if difference > ratio_threshold:
                                images_falling_below_or_large_ratio_count += 1
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    
    if total_images_count > 0:
        percentage = (images_falling_below_or_large_ratio_count / total_images_count) * 100
    else:
        percentage = 0
    
    print(f"Prozentanzahl der Bilder mit einer Abweichung > {ratio_threshold}% oder kleiner als {size_threshold}x{size_threshold}: {percentage:.2f}%")
    print(f"Gesamtanzahl der Bilder: {total_images_count}")
    print(f"Anzahl der Bilder, die die Bedingungen erfüllen: {images_falling_below_or_large_ratio_count}")
    
    return percentage, images_falling_below_or_large_ratio_count, total_images_count



def delete_images_falling_below_threshold_or_large_aspect_ratio_difference(folder_path, size_threshold=35, ratio_threshold=10):
    total_images_count = 0
    images_deleted_count = 0

    # iterate over all the images in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        total_images_count += 1  
                        
                        if width < size_threshold or height < size_threshold:
                            os.remove(file_path)
                            images_deleted_count += 1
                            print(f"Gelöscht: {file_path} (Größe: {width}x{height})")
                        else:
                            difference = abs(width - height) / max(width, height) * 100
                            
                            if difference > ratio_threshold:
                                os.remove(file_path)
                                images_deleted_count += 1
                                print(f"Gelöscht: {file_path} (Abweichung: {difference:.2f}%)")
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")
    
    print(f"Gesamtanzahl der Bilder: {total_images_count}")
    print(f"Anzahl der gelöschten Bilder: {images_deleted_count}")
    
    return images_deleted_count, total_images_count


def count_images_in_subfolders(folder_path):
    subfolder_image_counts = {}

    for root, dirs, files in os.walk(folder_path):
        image_count = 0
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_count += 1
        
        if image_count > 0:
            subfolder_image_counts[root] = image_count

    for subfolder, count in subfolder_image_counts.items():
        print(f"Unterordner: {subfolder}, Anzahl der Bilder: {count}")
    
    return subfolder_image_counts



def resize_image_with_largest_aspect_ratio_difference(folder_path, target_size=(35, 35), ratio_threshold=10):
    largest_difference = 0
    image_with_largest_difference = ""

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        if width != height:
                            difference = abs(width - height) / max(width, height) * 100
                            
                            if difference > largest_difference:
                                largest_difference = difference
                                image_with_largest_difference = file_path
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    if image_with_largest_difference:
        try:
            with Image.open(image_with_largest_difference) as img:
                original_path = f"{os.path.splitext(image_with_largest_difference)[0]}_original{os.path.splitext(image_with_largest_difference)[1]}"
                img.save(original_path)
                
                # resize image to 35*35px
                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_path = f"{os.path.splitext(image_with_largest_difference)[0]}_resized{os.path.splitext(image_with_largest_difference)[1]}"
                resized_img.save(resized_path)
                
                print(f"Originalbild gespeichert: {original_path}")
                print(f"Bild mit der größten Abweichung auf {target_size[0]}x{target_size[1]} resized: {resized_path}")
                print(f"Größter Seitenverhältnis-Unterschied: {largest_difference:.2f}%")
                print(f"Bild mit dem größten Unterschied: {image_with_largest_difference}")
        except Exception as e:
            print(f"Fehler beim Resizen oder Speichern des Bildes: {e}")
    else:
        print("Kein Bild mit ausreichend großer Abweichung gefunden.")
   
   
def resize_all_images_to_35x35(folder_path, target_size=(35, 35)):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # resize image to 35*35px
                        resized_img = img.resize(target_size, Image.LANCZOS)
                        
                        resized_img.save(file_path)
                        print(f"Bild resized und gespeichert: {file_path}")
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")   


def count_images_with_35x35_ratio(folder_path):
    count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('ppm', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if width == 35 and height == 35:
                            count += 1
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {file_path}: {e}")

    print(f"Anzahl der Bilder mit einer Größe von 35x35: {count}")
    return count
        

def clean_csv_files(folder_path):
    processed_files = 0

    if not os.path.exists(folder_path):
        print(f"Der Pfad '{folder_path}' existiert nicht.")
        return
    
    print(f"Durchsuche den Hauptordner: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        csv_file_path = None
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                print(f"Verarbeite CSV-Datei: {csv_file_path}")
                break  
            
        if csv_file_path is not None:
            # open CSV-file
            try:
                df = pd.read_csv(csv_file_path, delimiter=";")
            except Exception as e:
                print(f"Fehler beim Laden der CSV-Datei: {e}")
                continue
            
            rows_to_keep = []
            
            # check if the image files exist
            for _, row in df.iterrows():
                image_file_path = os.path.join(root, row["Filename"])
                if os.path.isfile(image_file_path):
                    rows_to_keep.append(row)
            
            # write the image files back to the CSV-file if updated
            if len(rows_to_keep) != len(df):
                updated_df = pd.DataFrame(rows_to_keep)
                updated_df.to_csv(csv_file_path, index=False, sep=";")
                print(f"Aktualisiert: {csv_file_path}")
            else:
                print(f"Keine Änderungen erforderlich für: {csv_file_path}")


            processed_files += 1
        else:
            print(f"Keine CSV-Datei in: {root}")

    print(f"Bereinigung abgeschlossen. {processed_files} CSV-Dateien verarbeitet.")


# example for use
folder_path = "../GTSRB/Final_Test/Images"

min_width = 35
min_height = 35

#unter 30x30: 4000
#unter 35x35: 10000
#unter 40x40: 17000

count_all_images(folder_path)
#count_small_images(folder_path, min_width, min_height)
#calculate_largest_aspect_ratio_difference(folder_path)
#calculate_percentage_of_images_with_large_aspect_ratio_difference(folder_path, threshold_percentage=10)
#count_images_falling_below_threshold_or_large_aspect_ratio_difference(folder_path, size_threshold=35, ratio_threshold=10)
#delete_images_falling_below_threshold_or_large_aspect_ratio_difference(folder_path, size_threshold=35, ratio_threshold=10)
#count_images_in_subfolders(folder_path)
#resize_image_with_largest_aspect_ratio_difference(folder_path, target_size=(35, 35), ratio_threshold=10)
#resize_all_images_to_35x35(folder_path, target_size=(35, 35))
count_images_with_35x35_ratio(folder_path)
# Funktion aufrufen mit dem gewünschten Pfad
#clean_csv_files(folder_path)