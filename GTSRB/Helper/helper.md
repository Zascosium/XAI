# Helper Guide: Skripte für die Vorverarbeitung von Bildern

In diesem Helper-Ordner haben wir mehrere Python-Skripte erstellt und genutzt, um unsere Bilddaten effizient vorzuverarbeiten. Die folgenden Skripte stehen zur Verfügung und erfüllen spezifische Aufgaben:

---

## 1. `rename_folder.py`

### Zweck:
Dieses Skript wird verwendet, um Ordner umzubenennen, um sicherzustellen, dass sie entsprechend des jeweiligen Straßenschilds benannt sind. 

### Nutzung:
- Das Skript durchläuft ein angegebenes Verzeichnis.
- Es benennt Ordner basierend auf der eingelesenen CSV-Datei. 

---

## 2. `put_test_data_in_folder.py`

### Zweck:
Dieses Skript dient dazu, Testdaten automatisch in die entsprechenden Ordner zu verschieben, um eine übersichtliche und strukturierte Datenorganisation zu gewährleisten.

### Nutzung:
- Testbilder werden basierend auf ihrem Label in die richtigen Ordner sortiert.

---

## 3. `images.py`

### Zweck:
Dieses Skript enthält Funktionen für die allgemeine Bildvorverarbeitung, wie z.B.:
- Skalierung
- Normalisierung
- Konvertierung zu 35x35

### Nutzung:
- Es wird verwendet, um Bilder für das CNN vorzubereiten.