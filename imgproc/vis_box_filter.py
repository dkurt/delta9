import argparse
import numpy as np
import re
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--input')
args = parser.parse_args()

def mapVal(x, y, shape, minVal=50, maxVal=200):
    ratio = float(x + y) / (shape[0] + shape[1] - 2)
    return maxVal - int((maxVal - minVal) * ratio)

producer = np.zeros((7, 7, 3), dtype=np.uint8)
consumer = np.zeros((7, 7, 3), dtype=np.uint8)

loadPoints = []

i = 0
with open(args.input, 'rt') as f:
    lines = f.read().rstrip('\n').split('\n')
    for line in lines:
        match = re.search('(Store|Load) (.*).*.0\((.*), (.*)\) = (.*)', line)
        if match:
            xs = [int(match.group(3))]
            ys = [int(match.group(4))]
        else:
            match = re.search('(Store|Load) (.*).*.0\(<(.*)>, <(.*)>\) = (.*)', line)
            if match:
                xs = [int(x) for x in match.group(3).split(', ')]
                ys = [int(y) for y in match.group(4).split(', ')]
        if match:
            mode = match.group(1)
            func = match.group(2)
            if func == 'producer':
                for x, y in zip(xs, ys):
                    producer[y + 1, x + 1] = [0, 255, 127] if mode == 'Store' else [0, 114, 255]
                    if mode == 'Load':
                        loadPoints.append((y + 1, x + 1))
            elif func == 'consumer':
                for x, y in zip(xs, ys):
                    consumer[y + 1, x + 1] = [0, 255, 127]
            else:
                assert (func == 'producer' or func == 'consumer')

            if mode == 'Store':
                out = np.concatenate((producer, consumer), axis=1)
                saved = cv.resize(out, (0, 0), fx=30, fy=30, interpolation=cv.INTER_NEAREST)

                cv.imshow('out', saved)
                # cv.imwrite('gifs/toucan_%06d.jpg' % i, saved)
                i += 1

                cv.waitKey()

                if func == 'consumer':
                    for x, y in zip(xs, ys):
                        consumer[y + 1, x + 1] = mapVal(x, y, consumer.shape)
                else:
                    for x, y in zip(xs, ys):
                        producer[y + 1, x + 1] = mapVal(x, y, producer.shape)

                for p in loadPoints:
                    producer[p[0], p[1]] = mapVal(p[1], p[0], producer.shape)
