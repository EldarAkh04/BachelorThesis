import pandas as pd
import os

# Pfade
excelPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/bildmetadaten.xlsx"
personsPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons"

# Excel laden (DeepFace-Klassifizierung)
df = pd.read_excel(excelPath)

# Dictionary: filename -> (race, gender) aus der Excel
excelData = {}
for _, row in df.iterrows():
    filename = str(row['filename']).strip()
    race = str(row['race']).strip().lower()
    gender = str(row['gender']).strip().lower()
    excelData[filename] = {'race': race, 'gender': gender}

# Durch alle Ordner laufen und tatsächliche Position ermitteln
mismatches = []
notInExcel = []
geprueft = 0

for root, dirs, files in os.walk(personsPath):
    for filename in files:
        if not filename.endswith(".png"):
            continue

        geprueft += 1

        # Tatsächliche Position aus dem Pfad ableiten
        # Struktur: .../Persons/{ethnie}/{geschlecht}/bild.png
        relPath = os.path.relpath(root, personsPath)
        teile = relPath.split(os.sep)

        if len(teile) < 2:
            # Bild liegt nicht in einer ethnie/geschlecht-Struktur
            continue

        ordnerEthnie = teile[0].strip().lower()
        ordnerGeschlecht = teile[1].strip().lower()

        # Mit Excel vergleichen
        if filename not in excelData:
            notInExcel.append(filename)
            continue

        excelEthnie = excelData[filename]['race']
        excelGeschlecht = excelData[filename]['gender']

        # Unterschiede finden
        ethnieDiff = ordnerEthnie != excelEthnie
        geschlechtDiff = ordnerGeschlecht != excelGeschlecht

        if ethnieDiff or geschlechtDiff:
            mismatches.append({
                'filename': filename,
                'ordner_ethnie': ordnerEthnie,
                'excel_ethnie': excelEthnie,
                'ethnie_unterschied': "JA" if ethnieDiff else "",
                'ordner_geschlecht': ordnerGeschlecht,
                'excel_geschlecht': excelGeschlecht,
                'geschlecht_unterschied': "JA" if geschlechtDiff else ""
            })

# Ausgabe
print("=" * 80)
print("KONTROLLE: Ordnerposition vs. DeepFace-Klassifizierung (Excel)")
print("=" * 80)
print(f"Geprüfte Bilder: {geprueft}")
print(f"Bilder mit abweichender Position (manuell korrigiert): {len(mismatches)}")
print(f"Bilder nicht in Excel gefunden: {len(notInExcel)}")

if mismatches:
    print("\n--- ABWEICHUNGEN ---")
    mismatchDf = pd.DataFrame(mismatches)
    print(mismatchDf.to_string(index=False))

    # Statistik
    ethnieKorrekturen = mismatchDf['ethnie_unterschied'].value_counts().get("JA", 0)
    geschlechtKorrekturen = mismatchDf['geschlecht_unterschied'].value_counts().get("JA", 0)
    print(f"\nEthnie-Korrekturen: {ethnieKorrekturen}")
    print(f"Geschlecht-Korrekturen: {geschlechtKorrekturen}")

if notInExcel:
    print("\n--- NICHT IN EXCEL GEFUNDEN ---")
    for f in notInExcel:
        print(f"  {f}")

# In Excel speichern
if mismatches:
    outputPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Positions_Kontrolle.xlsx"
    pd.DataFrame(mismatches).to_excel(outputPath, index=False)
    print(f"\nErgebnis gespeichert unter: {outputPath}")