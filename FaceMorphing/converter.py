from PIL import Image
import os

inputFolderPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/importJPG"
outputFolderPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/outputPNG"

filesInput = [f for f in os.listdir(inputFolderPath) if f.lower().endswith(".jpg")]

def convertAllToPNG(path1, path2):
        
    for i, filename in enumerate(filesInput, start=1):
        jpgImage = Image.open(os.path.join(path1, filename))
        jpgImage.save(os.path.join(path2, f"IM{i}.png"))
        print(f"Erfolgreich: {filename} -> IM{i}.png")

convertAllToPNG(inputFolderPath, outputFolderPath)
