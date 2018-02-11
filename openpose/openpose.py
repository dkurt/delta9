import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', dest='input', help='Path to imput image or video file')
parser.add_argument('--proto', dest='proto', help='Path to .prototxt')
parser.add_argument('--model', dest='model', help='Path to .caffemodel')
args = parser.parse_args()

BONES = [ [0, 1],                        # Head-Neck
          [1, 2], [2, 3], [3, 4],        # Neck-RShoulder-RElbow-RWrist
          [1, 5], [5, 6], [6, 7],        # Neck-LShoulder-LElbow-LWrist
          [1, 14],                       # Neck-Chest
          [14, 8], [8, 9], [9, 10],      # Chest-RHip-RKnee-RAnkle
          [14, 11], [11, 12], [12, 13]   # Chest-LHip-LKnee-LAnkle
]

net = cv.dnn.readNetFromCaffe(args.proto, args.model)
cap = cv.VideoCapture(args.input if args.input else 0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), False, False)
    net.setInput(blob)
    out = net.forward()

    points = []
    for i in range(out.shape[1]):
        heatMap = out[0, i]
        _, conf, _, point = cv.minMaxLoc(heatMap)

        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((x, y) if conf > 0.1 else None)

    for bone in BONES:
        idFrom = bone[0]
        idTo = bone[1]
        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    cv.imshow('OpenPose', frame)
