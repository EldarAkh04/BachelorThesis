#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

# ============================================================================
# ADAPTED BY: Eldar Akhundzada
# MODIFICATIONS (2026):
# - Converted to camelCase naming convention
# - Added custom functions: addExtraLandmarks(), addBorderPoints()
# - Added seamless cloning for background integration
# - Adapted for Face Morphing research [12]
# ============================================================================

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

import dlib
import os
import cv2
import numpy
from scipy.spatial import Delaunay

ShapePredictor = "shape_predictor_68_face_landmarks.dat"

imagePathA = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/Man/IM351.png"
imagePathB = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/Man/IM129.png"
predictor = dlib.shape_predictor(ShapePredictor)
detector = dlib.get_frontal_face_detector()

outputPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Results/"

def getLandMarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotateLandmarks(im, landmarks):
    im = im.copy()
    landmarks = numpy.array(landmarks)
    
    for idx in range(len(landmarks)):
        if landmarks.ndim == 2:
            x = int(landmarks[idx][0])
            y = int(landmarks[idx][1])
        else:
            x = int(landmarks[idx, 0])
            y = int(landmarks[idx, 1])
        
        pos = (x, y)
        
        cv2.putText(im, str(idx), pos, 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 2, color=(0, 255, 255), thickness=-1)
    return im

def transformationFromPoints(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warpIm(im, M, dshape):
    outputIm = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=outputIm,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return outputIm

def landmarksPos(landmarks):
    for i, point in enumerate(landmarks):
        x = int(landmarks[i, 0])
        y = int(landmarks[i, 1])
        print(f"Punkt {i} x={x}, y={y}")

def averageLandmark(landmarks1, landmarks2, alpha=0.5):
    landmarks1 = numpy.array(landmarks1).astype(float)
    landmarks2 = numpy.array(landmarks2).astype(float)
    
    landmarksMorphed = alpha * landmarks1 + alpha * landmarks2
    
    for i in range(len(landmarksMorphed)):
        x = int(landmarksMorphed[i, 0])
        y = int(landmarksMorphed[i, 1])
        print(f"Punkt {i}: x={x}, y={y}")
        
    return landmarksMorphed

def delaunayTriangle(landmarksMorphed):
    tri = Delaunay(landmarksMorphed)
    print(tri.simplices)
    return tri

def drawDelaunay(img, landmarks, simplices):
    img_copy = img.copy()
    landmarks = numpy.array(landmarks)
    
    for s in simplices:
        pt1 = tuple(landmarks[s[0]].astype(int))
        pt2 = tuple(landmarks[s[1]].astype(int))
        pt3 = tuple(landmarks[s[2]].astype(int))

        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img_copy, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)
        
    return img_copy

def morphTriangle(img1, img2, img, tri1, tri2, tri, alpha):
    r1 = cv2.boundingRect(numpy.float32([tri1]))
    r2 = cv2.boundingRect(numpy.float32([tri2]))
    r  = cv2.boundingRect(numpy.float32([tri]))

    t1Rect = []
    t2Rect = []
    tRect  = []

    for i in range(3):
        tRect.append(((tri[i][0] - r[0]), (tri[i][1] - r[1])))
        t1Rect.append(((tri1[i][0] - r1[0]), (tri1[i][1] - r1[1])))
        t2Rect.append(((tri2[i][0] - r2[0]), (tri2[i][1] - r2[1])))

    mask = numpy.zeros((r[3], r[2], 3), dtype=numpy.float32)
    cv2.fillConvexPoly(mask, numpy.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpMat1 = cv2.getAffineTransform(numpy.float32(t1Rect), numpy.float32(tRect))
    warpMat2 = cv2.getAffineTransform(numpy.float32(t2Rect), numpy.float32(tRect))

    imgRect1 = cv2.warpAffine(img1Rect, warpMat1, (size[0], size[1]), None, 
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    imgRect2 = cv2.warpAffine(img2Rect, warpMat2, (size[0], size[1]), None, 
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    imgRect = (1.0 - alpha) * imgRect1 + alpha * imgRect2
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask

def addExtraLandmarks(landmarks):
    landmarks = numpy.array(landmarks)
    extraPoints = []
    
    # 1. KIEFER (Punkte 0-16): Zwischen jedem Punkt einen hinzufügen
    for i in range(16):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    
    # 2. AUGENBRAUEN (17-21 rechts, 22-26 links)
    # Rechte Augenbraue
    for i in range(17, 21):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    
    # Linke Augenbraue
    for i in range(22, 26):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    
    # 3. NASE (27-35)
    # Nasenrücken
    for i in range(27, 30):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    
    # Nasenflügel
    for i in range(31, 35):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    
    # 4. AUGEN (36-41 rechts, 42-47 links)
    # Rechtes Auge
    for i in range(36, 41):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    # Schließe rechtes Auge
    mid = (landmarks[41] + landmarks[36]) / 2
    extraPoints.append(mid)
    
    # Linkes Auge
    for i in range(42, 47):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    # Schließe linkes Auge
    mid = (landmarks[47] + landmarks[42]) / 2
    extraPoints.append(mid)
    
    # 5. MUND AUSSEN (48-59)
    for i in range(48, 59):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    # Schließe äußeren Mund
    mid = (landmarks[59] + landmarks[48]) / 2
    extraPoints.append(mid)
    
    # 6. MUND INNEN (60-67)
    for i in range(60, 67):
        p1 = landmarks[i]
        p2 = landmarks[i + 1]
        mid = (p1 + p2) / 2
        extraPoints.append(mid)
    # Schließe inneren Mund
    mid = (landmarks[67] + landmarks[60]) / 2
    extraPoints.append(mid)
    
    # Stirn (oberhalb der Augenbrauen)
    eyebrow_center_right = (landmarks[19] + landmarks[20]) / 2
    eyebrow_center_left = (landmarks[23] + landmarks[24]) / 2
    nose_top = landmarks[27]
    
    # Stirnpunkte
    forehead_right = eyebrow_center_right + numpy.array([0, -30])
    forehead_left = eyebrow_center_left + numpy.array([0, -30])
    forehead_center = nose_top + numpy.array([0, -50])
    
    extraPoints.extend([forehead_right, forehead_left, forehead_center])
    
    # Wangenpunkte (zwischen Auge und Kiefer)
    # Rechte Wange
    cheekRight1 = (landmarks[2] + landmarks[36]) / 2
    cheekRight2 = (landmarks[3] + landmarks[39]) / 2
    cheekRight3 = (landmarks[4] + landmarks[31]) / 2
    
    # Linke Wange
    cheekLeft1 = (landmarks[14] + landmarks[45]) / 2
    cheekLeft2 = (landmarks[13] + landmarks[42]) / 2
    cheekLeft3 = (landmarks[12] + landmarks[35]) / 2
    
    extraPoints.extend([
        cheekRight1, cheekRight2, cheekRight3,
        cheekLeft1, cheekLeft2, cheekLeft3
    ])
    
    # Kinnpunkte (zusätzliche Punkte am Kinn)
    chinCenter = landmarks[8]
    chinLeft = (landmarks[6] + landmarks[8]) / 2
    chinRight = (landmarks[10] + landmarks[8]) / 2
    
    extraPoints.extend([chinLeft, chinCenter ,chinRight])
    
    
    allLandmarks = numpy.vstack([landmarks, extraPoints])
    
    return allLandmarks

def addBorderPoints(landmarks, imgShape):
    h, w = imgShape[:2]
    landmarks = numpy.array(landmarks)
    
    border = [
        [0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
        [w//2, 0], [0, h//2], [w-1, h//2], [w//2, h-1],
        [w//4, 0], [3*w//4, 0],
        [w//4, h-1], [3*w//4, h-1],
        [0, h//4], [0, 3*h//4],
        [w-1, h//4], [w-1, 3*h//4]
    ]
    
    return numpy.vstack([landmarks, border])

imageA = cv2.imread(imagePathA)
imageB = cv2.imread(imagePathB)

landmarksA = getLandMarks(imageA)
landmarksB = getLandMarks(imageB)


#Affine Transformation
M = transformationFromPoints(landmarksA, landmarksB)
imageBaligned = warpIm(imageB, M, imageA.shape)
alignedLandmarksB = getLandMarks(imageBaligned)

landmarksAExtra = addExtraLandmarks(landmarksA)
landmarksBExtra = addExtraLandmarks(alignedLandmarksB)

landmarksAFull = addBorderPoints(landmarksAExtra, imageA.shape)
landmarksBFull = addBorderPoints(landmarksBExtra, imageBaligned.shape)

landmarksAFull = numpy.matrix(landmarksAFull)
landmarksBFull = numpy.matrix(landmarksBFull)

print(f"Original Landmarks: 68")
print(f"Mit Extra Landmarks: {len(landmarksAExtra)}")
print(f"Mit Randpunkten: {len(landmarksAFull)}")

imageWithLandMarksA = annotateLandmarks(imageA, landmarksAFull)
imageWithLandMarksB = annotateLandmarks(imageBaligned, landmarksBFull)

cv2.imwrite(outputPath + 'ImageA.png', imageWithLandMarksA)
cv2.imwrite(outputPath + 'ImageB.png', imageWithLandMarksB)

""" cv2.imshow('ImageA', imageWithLandMarksA)
cv2.imshow('ImageB', imageWithLandMarksB)

cv2.waitKey(0)
cv2.destroyAllWindows() """

koordinationA = landmarksPos(landmarksAFull)
print("-"*30)
koordinationB = landmarksPos(landmarksBFull)
print("-"*30)

# Morphed Landmarks mit jedem Punkten
landmarksMorphed = averageLandmark(landmarksAFull, landmarksBFull)
# Delaunay Triangulation:
tri = delaunayTriangle(landmarksMorphed)
blankImage = numpy.zeros(imageA.shape, dtype=numpy.uint8)
debugImage = drawDelaunay(blankImage, landmarksMorphed, tri.simplices)
cv2.imwrite(outputPath + "DelaunayTrian.png", debugImage)
print(f"Anzahl Dreiecke: {len(tri.simplices)}")


debugImageA = imageA.copy()
debugImageA = drawDelaunay(debugImageA, landmarksAFull, tri.simplices)
cv2.imwrite(outputPath + "DelaunayTrian_A.png", debugImageA)

debugImageB = imageBaligned.copy()
debugImageB = drawDelaunay(debugImageB, landmarksBFull, tri.simplices)
cv2.imwrite(outputPath + "DelaunayTrian_B.png", debugImageB)

# Morphing mit jedem Landmark
imgMorph = numpy.zeros(imageA.shape, dtype=imageA.dtype)
for s in tri.simplices:
    t1 = numpy.array([landmarksAFull[s[0]], landmarksAFull[s[1]], landmarksAFull[s[2]]], dtype=numpy.float32).reshape(3, 2)
    t2 = numpy.array([landmarksBFull[s[0]], landmarksBFull[s[1]], landmarksBFull[s[2]]], dtype=numpy.float32).reshape(3, 2)
    t  = numpy.array([landmarksMorphed[s[0]], landmarksMorphed[s[1]], landmarksMorphed[s[2]]], dtype=numpy.float32).reshape(3, 2)

    morphTriangle(imageA, imageBaligned, imgMorph, t1, t2, t, 0.5)

imgMorph = cv2.bilateralFilter(imgMorph, 5, 50, 50)

cv2.imwrite(outputPath + "FinalMorph.png", imgMorph)
print("Morph gespeichert!")

#Hintergrund übernehmen:
morphedPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Results/FinalMorph.png"
if(os.path.exists(morphedPath)):
    print("ist vorhanden")

imageM = cv2.imread(morphedPath)
landmarksM = getLandMarks(imageM)

M_M = transformationFromPoints(landmarksA, landmarksM)
imageMaligned = warpIm(imageM, M_M, imageA.shape)
alignedLandmarksM = getLandMarks(imageMaligned)

imageWithLandMarksA = annotateLandmarks(imageA, landmarksA)
imageWithLandMarksM = annotateLandmarks(imageMaligned, alignedLandmarksM)

""" cv2.imshow('ImageA', imageWithLandMarksA)
cv2.imshow('Morphed Image mit Landmarks', imageWithLandMarksM)

cv2.imwrite('ImageA.png', imageWithLandMarksA)
cv2.imwrite('ImageM.png', imageWithLandMarksM)

cv2.waitKey(0)
cv2.destroyAllWindows() """
#Hintergrund
src = imageMaligned
dst = imageA

mask = numpy.zeros(dst.shape[:2], dtype=numpy.uint8)
outerPoints = list(range(0, 27))
hull = cv2.convexHull(numpy.array(landmarksA[outerPoints], dtype=numpy.int32))
cv2.fillPoly(mask, [hull], 255)

kernel = numpy.ones((15, 15), numpy.uint8)
maskEroded = cv2.erode(mask, kernel, iterations=1)

maskSoft = cv2.GaussianBlur(maskEroded, (71, 71), 0)

alpha = cv2.cvtColor(maskSoft, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
srcBlended = (src.astype(float) * alpha + dst.astype(float) * (1.0 - alpha)).astype(numpy.uint8)

r = cv2.boundingRect(hull)
center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

output = cv2.seamlessClone(srcBlended, dst, maskEroded, center, cv2.NORMAL_CLONE)

cv2.imwrite("/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/Persons/latino hispanic/Man/129/IM129-hig.png", output)