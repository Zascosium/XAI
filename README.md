# XAI: Die Auswirkungen von Optimierungsalgorithmen auf die Feature Map eines CNN
Wir haben die README-Datei, die Datei helper.md unter "GTSRB/Helper/helper.md" sowie Kommentare an wichtigen Stellen oder dort, wo sie sinnvoll sind, hinzugefügt, um den Code verständlicher zu machen. Zusätzlich wurden Markdown-Felder in der Datei main.ipynb eingefügt, um die grundlegende Struktur des Codes zu verbessern.

## Projektbeschreibung
In diesem Projekt untersuchen wir die Auswirkungen verschiedener Optimierungsalgorithmen auf die Feature Maps eines Convolutional Neural Networks (CNN). Zusätzlich verwenden wir Explainable AI (XAI)-Methoden wie GradCAM und Lime, um die Entscheidungsprozesse des Modells zu visualisieren. Das verwendete Dataset ist das **German Traffic Sign Recognition Benchmark (GTSRB)**, welches aus 43 Klassen besteht.

## Datensatz: GTSRB
- **Name**: German Traffic Sign Recognition Benchmark
- **Problemtyp**: Single-Image, Multi-Klassen-Klassifikation (43 Klassen)
- **Anzahl der Bilder**: Mehr als 50.000
- **Klassenübersicht**: Die folgende Tabelle zeigt alle 43 Klassen des Datensatzes:

| Index | Label                              |
|-------|------------------------------------|
| 0     | 100_kmh                            |
| 1     | 120_kmh                            |
| 2     | 20_kmh                             |
| 3     | 30_kmh                             |
| 4     | 50_kmh                             |
| 5     | 60_kmh                             |
| 6     | 70_kmh                             |
| 7     | 80_aufgehoben                      |
| 8     | 80_kmh                             |
| 9     | achtung_eis                        |
| 10    | achtung_fahrrad                    |
| 11    | achtung_kinder                     |
| 12    | achtung_wildwechsel                |
| 13    | ampel                              |
| 14    | baustelle                          |
| 15    | durchfahrt_verboten                |
| 16    | einmalige_vorfahrt                 |
| 17    | fussgaenger                        |
| 18    | gefahrenstelle                     |
| 19    | kreisverkehr                       |
| 20    | lkw_ueberholverbot                 |
| 21    | lkw_ueberholverbot_aufgehoben      |
| 22    | lkw_verboten                       |
| 23    | rutschgefahr                       |
| 24    | s_kurve_links                      |
| 25    | scharfe_kurve_links                |
| 26    | scharfe_kurve_rechts               |
| 27    | stopp                              |
| 28    | ueberholverbot                     |
| 29    | ueberholverbot_aufgehoben          |
| 30    | unbegrenzt                         |
| 31    | unebene_fahrbahn                   |
| 32    | verbot_der_einfahrt                |
| 33    | verengte_fahrbahn                  |
| 34    | vorbeifahren_links                 |
| 35    | vorbeifahren_rechts                |
| 36    | vorfahrsstrasse                    |
| 37    | vorfahrt_gewaehren                 |
| 38    | vorgeschrieben_geradeaus           |
| 39    | vorgeschrieben_geradeaus_links     |
| 40    | vorgeschrieben_geradeaus_rechts    |
| 41    | vorgeschrieben_links               |
| 42    | vorgeschrieben_rechts              |

---

## Datenvorbereitung
Die Datenvorbereitung besteht aus den folgenden Schritten:
1. **Filtern**: Unbrauchbare oder defekte Daten werden entfernt. (Bildgröße unter 35x35 und mit mehr als 10% Abweichung im Seitenverhältnis)
2. **Skalieren**: Bilder werden auf einheitliche Dimensionen skaliert (35x35)
3. **Ordnerstruktur**: Die Daten werden in eine geeignete Ordnerstruktur für Trainings- und Testphasen aufgeteilt. (Pro Klasse ein Ordner)
4. **Augmentation**: Zur Verbesserung der Generalisierung werden Techniken wie Rotation, Zoom und Farbverschiebung angewendet. (Rotation, Resize, Änderung der Helligkeit/Kontrast/Sättigung/Farbton, Spiegelungen)

## Optimierungsalgorithmen
Folgende Optimierungsalgorithmen wurden getestet:
- **Adam**
- **SGD (Stochastic Gradient Descent)**
- **Adagrad**
- **RMSprop**
- **SGD (schlecht konfiguriert)**

---

## Verwendete Methoden
Wir haben folgende Methoden zur Visualisierung und Erklärung der Ergebnisse verwendet:

### 1. GradCAM
- **Ziel**: Visualisierung von Aktivierungen im CNN.
- **Beschreibung**: Es werden Heatmaps erstellt, die die Bildbereiche hervorheben, die das Modell für seine Entscheidungen verwendet.

### 2. LIME (Local Interpretable Model-Agnostic Explanations)
- **Ziel**: Lokale Erklärung der Modellentscheidungen.
- **Beschreibung**: LIME segmentiert Bilder (z. B. mit dem SLIC-Algorithmus) und untersucht die Relevanz einzelner Segmente für die Klassifikation.

### 3. Aktivierung der Convolutional Layer (Feature Maps)
- **Ziel**: Untersuchung der Aktivierungsmuster in Convolutional Layers.
- **Beschreibung**: Die Feature Maps der Convolutional Layers werden visualisiert, um zu analysieren, welche Merkmale in den Schichten extrahiert werden.

### 4. Aktivierung der Fully Connected Layer (Letzte Schicht)
- **Ziel**: Analyse der Neuronenaktivierung in der letzten Fully Connected Layer, um zu verstehen, welche Neuronen (entsprechend den 43 Klassen) wie stark aktiviert werden.
- **Beschreibung**: Für jeden Output (entsprechend einer der 43 Klassen) wird die Aktivierung jedes Neurons visualisiert.

### 5. Accuracy pro Klasse und Missklassifikation
- **Ziel**: Bewertung der Modellleistung auf Klassenebene.
- **Beschreibung**: Für jede Klasse wird die Genauigkeit berechnet, und Missklassifikationen werden visualisiert.

### 6. Maximale Aktivierung eines Neurons
- **Ziel**: Untersuchung der spezifischen Aktivierung eines Neurons.
- **Beschreibung**: Es wird analysiert, welches Eingabebild eine maximale Aktivierung für ein bestimmtes Neuron auslöst. Dies liefert Einsichten in die spezifischen Merkmale, auf die das Neuron reagiert.
---

## Installation und Nutzung

1. Klone das Repository:
   ```bash
   git clone https://github.com/Zascosium/XAI.git
   cd XAI
2. Installiere die requirements
    ```bash
    pip install -r requirements.txt
3. (optional) Cuda Installation
    ```bash
    torch==<version>+cu11x