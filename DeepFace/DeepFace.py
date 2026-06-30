from deepface import DeepFace
import json
import pandas as pd
import os


img1 = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/indian/Man/IM190.png"
img2 = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/indian/Man/IM181.png"

imgMorph = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/indian/Man/190/IM190-low.png"
#imgMorph = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/TestE.png"

folderPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic"

compareImage = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/black/Man/IM471.png"
pathToCompare = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/black/Man"

def passControl(img1, img2):
    result = DeepFace.verify(img1, img2, model_name = "Facenet512")
    #result = DeepFace.verify(img1, img2, model_name = "VGG-Face")
    verified = bool(result['verified'])
    distance = round(result['distance'], 4)
    threshold = float(result['threshold'])
    print("Ähnlichkeit: " + result['similarity_metric'])

    print("Verified: " + str(verified))
    print("Distanz: " + str(distance))
    print("Threshold: " + str(threshold))

def classifyPerson(folderPath):
    data = []
    for root, dirs, files in os.walk(folderPath):
        for filename in files:
            if filename.endswith(".png"):
                try:
                    imgPath = os.path.join(root, filename)
                    
                    analysis = DeepFace.analyze(imgPath, actions=['age', 'gender', 'race'])[0]
                    
                    data.append({
                        'filename': filename,
                        'folder': os.path.basename(root),
                        'age': analysis['age'],
                        'gender': analysis['dominant_gender'],
                        'race': analysis['dominant_race']
                    })
                    
                    print(f"✅ {filename}")
                    
                except Exception as e:
                    print(f"Fehler: {filename}")
    
    df = pd.DataFrame(data)
    df.to_excel("/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/bildmetadatenLatino.xlsx", index=False)
    
    print(f"Fertig: {len(data)} Bilder analysiert")

def comparePersons(compareImage, pathToCompare): 
    imageFiles = [f for f in os.listdir(pathToCompare) if f.endswith(".png")]
    data = []
    
    for filename in imageFiles:
        imgPath = os.path.join(pathToCompare, filename)
        result = DeepFace.verify(compareImage, imgPath, model_name="Facenet512")
        distance = round(result['distance'], 4)
        analysis = DeepFace.analyze(imgPath, actions=['age'])[0]
        print("Distanz: " + str(distance) + " " + f"Bild: {filename}" + " " + "Age: " + str(analysis['age']))
        data.append({
            'filename': filename,
            'age' : analysis['age'],
            'distance': distance      
        })   
    
    df = pd.DataFrame(data)
    df.to_excel("/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/black/Man/vergleiche471.xlsx", index=False)
    print("Fertig")

#comparePersons(compareImage, pathToCompare)
passControl(img1, imgMorph)
#classifyPerson(folderPath)