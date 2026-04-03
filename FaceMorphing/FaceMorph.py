import dlib
import os
import cv2
import numpy

ShapePredictor = "shape_predictor_68_face_landmarks.dat"
imagePath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Trump.png"
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


image = cv2.imread(imagePath)
landmarks = getLandMarks(image)
imageWithLandMarks = annotateLandmarks(image, landmarks)

cv2.imshow('Result', imageWithLandMarks)
cv2.imwrite('Trump.png', imageWithLandMarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
