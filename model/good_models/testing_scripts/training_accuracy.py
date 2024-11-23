import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=4, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=4, padding=1)
        self.relu3 = nn.ReLU()

        self.conv_layer4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=4, padding=1)
        self.relu4 = nn.ReLU()

        self.conv_layer5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(128, 512) 
        self.relu6 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()

        self.dropout8 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(256, 128)
        self.relu8 = nn.ReLU()

        self.fc9 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)

        out = self.conv_layer2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)

        out = self.conv_layer3(out)
        out = self.relu3(out)

        out = self.conv_layer4(out)
        out = self.relu4(out)

        out = self.conv_layer5(out)
        out = self.relu5(out)
        out = self.max_pool5(out)

        out = out.reshape(out.size(0), -1)
        out = self.dropout6(out)
        out = self.fc6(out)
        out = self.relu6(out)

        out = self.dropout7(out)
        out = self.fc7(out)
        out = self.relu7(out)

        out = self.dropout8(out)
        out = self.fc8(out)
        out = self.relu8(out)

        out = self.fc9(out)
        return out

# Prüfen, ob das Modell geladen werden kann und die Pfade korrekt sind
model_path = 'model/base_sgd_model.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Das Modell unter '{model_path}' wurde nicht gefunden.")

# Modell laden
model = CNN(num_classes=43)
model.load_state_dict(torch.load(model_path))
model.eval()

# Bildvorverarbeitung
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((35, 35)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# Vorhersage für ein Bild
def predict(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Test auf dem gesamten Datensatz
def test_model(train_dir):
    correct = 0
    total = 0
    
    # Durch alle Klassenordner iterieren, die jetzt den Klassennamen als Ordnernamen haben
    for class_folder_name in os.listdir(train_dir):
        class_folder = os.path.join(train_dir, class_folder_name)
        
        if not os.path.isdir(class_folder):
            continue

        try:
            # Klassennummer aus dem Ordnernamen extrahieren
            class_id = int(class_folder_name)
        except ValueError:
            print(f"Ordnername '{class_folder_name}' ist keine gültige Klassennummer.")
            continue

        csv_path = os.path.join(class_folder, f"GT-{class_folder_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"CSV-Datei nicht gefunden: {csv_path}")
            continue
        
        # CSV laden
        train_data = pd.read_csv(csv_path, sep=';')
        
        # Vorhersagen für jedes Bild in der CSV
        for _, row in train_data.iterrows():
            filename = row['Filename']
            image_path = os.path.join(class_folder, filename)
            if not os.path.exists(image_path):
                print(f"Bild nicht gefunden: {image_path}")
                continue

            predicted_class = predict(image_path)
            
            if predicted_class == class_id:
                correct += 1
            else:
                print(f"Falsche Vorhersage für {image_path}: {predicted_class} (sollte {class_id} sein)")
            total += 1
    
    # Genauigkeit berechnen
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"Genauigkeit des Modells auf dem Trainingsdatensatz: {accuracy:.2f}%")

# Beispiel für die Verwendung
train_dir = 'GTSRB/Final_Training/Images'
test_model(train_dir)
