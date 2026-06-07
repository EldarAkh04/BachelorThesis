from PIL import Image
import os

inputFolderPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/importJPG"
outputFolderPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/outputPNG"

def convertAllToPNG(inputPath, outputPath):
    counter = 1
    
    for root, dirs, files in os.walk(inputPath):
        for filename in files:
            #if filename.lower().endswith(".jpg"):
            if filename.lower().endswith(".jpg") and "-N" in filename:
                jpgPath = os.path.join(root, filename)
                jpgImage = Image.open(jpgPath)
                
                outputFileName = f"IM{counter}.png"
                jpgImage.save(os.path.join(outputPath, outputFileName))
                
                print(f"Erfolgreich: {filename} -> {outputFileName}")
                counter += 1

convertAllToPNG(inputFolderPath, outputFolderPath)
