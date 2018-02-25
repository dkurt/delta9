import argparse
import numpy as np
import re
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--input')
args = parser.parse_args()

def mapVal(x, y, shape, minVal=50, maxVal=200):
    ratio = float(x + y) / (shape[0] + shape[1] - 2)
    return minVal + int((maxVal - minVal) * ratio)

out = np.zeros((10, 10, 3), dtype=np.uint8)
i = 0
with open(args.input, 'rt') as f:
    lines = f.read().rstrip('\n').split('\n')
    for line in lines:
        match = re.search('.*\((\d*), (\d*)\) = (\d*)', line)
        if match:
            xs = [int(match.group(1))]
            ys = [int(match.group(2))]
        else:
            match = re.search('.*\(<(.*)>, <(.*)>\) = (.*)', line)
            if match:
                xs = [int(x) for x in match.group(1).split(', ')]
                ys = [int(y) for y in match.group(2).split(', ')]

        if match:
            for x, y in zip(xs, ys):
                out[y, x] = [0, 255, 127]
            saved = cv.resize(out, (300, 300), interpolation=cv.INTER_NEAREST)
            cv.imwrite('gifs/toucan_%06d.jpg' % i, saved)
            i += 1
            cv.imshow('out', saved)
            for x, y in zip(xs, ys):
                out[y, x] = mapVal(x, y, out.shape)
            # cv.waitKey(100)
