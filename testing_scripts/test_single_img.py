import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
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

        self.conv_layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.dropout6 = nn.Dropout(p=0.5)
        
        
        self.fc6 = nn.Linear(256, 512) 
        self.relu6 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(512, 256)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(256, num_classes)

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

        out = self.fc8(out)

        return out


# Prüfe, ob das Modell geladen werden kann und die Pfade korrekt sind
model_path = 'model/model.pth'  # Pfad zum Modell
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Das Modell unter '{model_path}' wurde nicht gefunden.")

# 1. Laden des Modells
model = CNN(num_classes=43)  # Modell initialisieren
model.load_state_dict(torch.load('model/model.pth'))  # Gewichte laden
model.eval() # Setzt das Modell in den Evaluationsmodus

# 2. Bildvorverarbeitung
def preprocess_image(image_path):
    # Bild öffnen und in RGB konvertieren
    image = Image.open(image_path).convert("RGB")
    
    # Transformations: Anpassung an das Modell (z.B. Größe, Normalisierung)
    transform = transforms.Compose([
        transforms.Resize((35, 35)),  # Beispiel für Resizing, falls Modell 224x224 erwartet
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Angepasste Normwerte für RGB-Bilder
    ])
    
    return transform(image).unsqueeze(0)  # Hinzufügen einer Batch-Dimension

# 3. Vorhersagefunktion
def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Das Bild unter '{image_path}' wurde nicht gefunden.")
    
    # Bild vorverarbeiten
    input_tensor = preprocess_image(image_path)
    
    # Mit dem Modell Vorhersagen machen
    with torch.no_grad():
        output = model(input_tensor)
    
    # Index des höchsten Wertes als vorhergesagte Klasse
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Beispiel für die Verwendung
image_path = 'vorfahrt.jpg'  # Pfad zum Bild
result = predict(image_path)
print("Vorhergesagte Klasse:", result)