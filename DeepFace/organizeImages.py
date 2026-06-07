import pandas as pd
import os
import shutil

excelPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/bildmetadatenLatino.xlsx"
sourceFolder = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic"

targetBase = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic"

genderM = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/Man"
genderW = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/Woman"


def organizeByEthnic(excelPath, sourceFol, targetBase):
    df = pd.read_excel(excelPath)
    ethnicities = df['race'].unique()

    for ethnicity in ethnicities:
        ethnicFolder = os.path.join(targetBase, ethnicity)
        os.makedirs(ethnicFolder, exist_ok=True)

    moved = 0
    for index, row in df.iterrows():
        filename = row['filename']
        ethnicity = row['race']
        
        sourcePath = os.path.join(sourceFol, filename)
        targetPath = os.path.join(targetBase, ethnicity, filename)
        
        try:
            shutil.move(sourcePath, targetPath)
            moved += 1
            
            if moved % 100 == 0:
                print(f"{moved}/{len(df)} verschoben...")
                
        except Exception as e:
            print(f"Fehler: {filename}")
    
    print(f"Fertig: {moved} Bilder verschoben")



def organizeByGender(excelPath, sourceFolder, targetM, targetW):
    df = pd.read_excel(excelPath)
    os.makedirs(targetM, exist_ok=True)
    os.makedirs(targetW, exist_ok=True)
    
    moved = 0
    
    for index, row in df.iterrows():
        filename = row['filename']
        gender = row['gender']
        
        sourcePath = os.path.join(sourceFolder, filename)
        if gender == 'Man':
            targetPath = os.path.join(targetM, filename)
        else:
            targetPath = os.path.join(targetW, filename)
        
        try:
            shutil.move(sourcePath, targetPath)
            moved += 1
        except Exception as e:
            print(f"Fehler: {filename} - {e}")
    
    print(f"Fertig: {moved} Bilder verschoben")


#organizeByEthnic(excelPath, sourceFolder, targetBase)
organizeByGender(excelPath, sourceFolder, genderM, genderW)