import dlib
import os
import cv2
import numpy
from scipy.spatial import Delaunay

ShapePredictor = "shape_predictor_68_face_landmarks.dat"
imagePathA = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Image1.png"
imagePathB = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/Image2.png"
predictor = dlib.shape_predictor(ShapePredictor)
detector = dlib.get_frontal_face_detector()

#https://github.com/matthewearl/faceswap/blob/master/faceswap.py 
#Wurde eigentlich für faceswap benutzt: http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
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

def delaunayTriangle(landmarksMorphed):
    tri = Delaunay(landmarksMorphed)
    print(tri.simplices)
    return tri

def drawDelaunay(img, landmarks, simplices):
    img_copy = img.copy()
    
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

imageA = cv2.imread(imagePathA)
imageB = cv2.imread(imagePathB)

landmarksA = getLandMarks(imageA)
landmarksB = getLandMarks(imageB)

M = transformationFromPoints(landmarksA, landmarksB)
imageBaligned = warpIm(imageB, M, imageA.shape)
alignedLandmarksB = getLandMarks(imageBaligned)

imageWithLandMarksA = annotateLandmarks(imageA, landmarksA)
imageWithLandMarksB = annotateLandmarks(imageBaligned, alignedLandmarksB)

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

landmarksMorphed = averageLandmark(landmarksA, landmarksB)
tri = delaunayTriangle(landmarksMorphed)
blank_image = numpy.zeros(imageA.shape, dtype=numpy.uint8)
debugImage = drawDelaunay(blank_image, landmarksMorphed, tri.simplices)
cv2.imwrite("DelaunayCheckPure.png", debugImage)
imgMorph = numpy.zeros(imageA.shape, dtype=imageA.dtype)
for s in tri.simplices:
    t1 = numpy.array([landmarksA[s[0]], landmarksA[s[1]], landmarksA[s[2]]], dtype=numpy.float32).reshape(3, 2)
    t2 = numpy.array([alignedLandmarksB[s[0]], alignedLandmarksB[s[1]], alignedLandmarksB[s[2]]], dtype=numpy.float32).reshape(3, 2)
    t  = numpy.array([landmarksMorphed[s[0]], landmarksMorphed[s[1]], landmarksMorphed[s[2]]], dtype=numpy.float32).reshape(3, 2)

    morphTriangle(imageA, imageBaligned, imgMorph, t1, t2, t, 0.5)
cv2.imwrite("FinalMorph.png", imgMorph)

#HIntergrund übernehmen:
morphedPath = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/FaceMorphing/FinalMorph.png"
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
#Hintergrund: https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/
src = imageMaligned 
dst = imageA        

srcMask = numpy.zeros(dst.shape, dst.dtype)
outerPoints = list(range(0, 27)) 

hull = cv2.convexHull(numpy.array(landmarksA[outerPoints], dtype=numpy.int32))
cv2.fillPoly(srcMask, [hull], (255, 255, 255))

r = cv2.boundingRect(hull)
center = (r[0] + r[2] // 2, r[1] + r[3] // 2)

output = cv2.seamlessClone(src, dst, srcMask, center, cv2.NORMAL_CLONE)

cv2.imwrite("morphWithBackgroundOriginal.png", output)
