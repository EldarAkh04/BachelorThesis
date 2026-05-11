from deepface import DeepFace
import json


img1 = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/TestIMG/IM17.png"
img2 = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/TestIMG/IM7.png"



def passControl(img1, img2):
    result = DeepFace.verify(img1, img2, model_name = "Facenet512")
    obj = DeepFace.analyze(img1, actions=['age','gender', 'race'])[0]

    print("Geschlecht: " +  obj['dominant_gender'])
    age = int(obj['age'])
    print("Alter: " +  str(age))
    print("Ethnizität: " + obj['dominant_race'])
    verified = bool(result['verified'])
    distance = round(result['distance'], 4)
    threshold = float(result['threshold'])

    print("Verified: " + str(verified))
    print("Distanz: " + str(distance))
    print("Treshold: " + str(threshold))
    print("Ähnlichkeit: " + result['similarity_metric'])

passControl(img1, img2)