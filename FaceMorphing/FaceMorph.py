import dlib
import os
import cv2
import numpy

ShapePredictor = "shape_predictor_68_face_landmarks.dat"
imagePathA = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Image1.png"
imagePathB = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Image2.png"
predictor = dlib.shape_predictor(ShapePredictor)
detector = dlib.get_frontal_face_detector()

def getLandMarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotateLandmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
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
    
    landmarksMorphed = (1 - alpha) * landmarks1 + alpha * landmarks2
    
    for i in range(len(landmarksMorphed)):
        x = int(landmarksMorphed[i, 0])
        y = int(landmarksMorphed[i, 1])
        print(f"Punkt {i}: x={x}, y={y}")
        
    return landmarksMorphed


imageA = cv2.imread(imagePathA)
imageB = cv2.imread(imagePathB)

landmarksA = getLandMarks(imageA)
landmarksB = getLandMarks(imageB)

M = transformationFromPoints(landmarksA, landmarksB)
imageB_aligned = warpIm(imageB, M, imageA.shape)
alignedLandmarksB = getLandMarks(imageB_aligned)

imageWithLandMarksA = annotateLandmarks(imageA, landmarksA)
imageWithLandMarksB = annotateLandmarks(imageB_aligned, alignedLandmarksB)

""" cv2.imshow('ImageA', imageWithLandMarksA)
cv2.imshow('ImageB', imageWithLandMarksB)

cv2.imwrite('ImageA.png', imageWithLandMarksA)
cv2.imwrite('ImageB.png', imageWithLandMarksB)

cv2.waitKey(0)
cv2.destroyAllWindows() """

koordinationA = landmarksPos(landmarksA)
print("-"*30)
koordinationB = landmarksPos(landmarksB)
print("-"*30)
averageLandmark(landmarksA, landmarksB)



