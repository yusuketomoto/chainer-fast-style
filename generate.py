import numpy as np
import argparse
from PIL import Image
import time, os, glob
import cv2, re

import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--scale', '-x', default=1, type=float)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--optical_flow', '-flow', default=0, type=float)
parser.add_argument('--original_colors', dest='ocol', action='store_true')
parser.set_defaults(ocol=False)
args = parser.parse_args()

def alnumlist(s):
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', s)]

def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

def blend(img1, img2, alpha):
    if not isinstance(alpha, float):
        alpha = np.dstack((alpha, alpha, alpha))
    return ((1-alpha)*img1) + (alpha*img2)

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

result = []
imagelist = sorted(glob.glob(args.input), key=alnumlist)
multi = True if len(imagelist) > 1 else False
for i, filename in enumerate(imagelist):
    start = time.time()
    image = Image.open(filename).convert('RGB')
    if args.optical_flow and len(result):
        img1 = cv2.imread(imagelist[i-1], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img2color = np.asarray(image)[:, :, ::-1]
        img1styled = result[:, :, ::-1]
        img1styled = cv2.resize(img1styled, (img1.shape[1], img1.shape[0]))
        
        print "Calculating flow",
        flow = cv2.calcOpticalFlowFarneback(img1, img2, 0.5, 3, 15, 3, 5, 1.2, 0)
        ys = range(flow.shape[0])
        xs = range(flow.shape[1])
        mapping = flow.copy()
        for y in ys:
            mapping[y, :, 0] = xs - flow[y, :, 0]
        for x in xs:
            mapping[:, x, 1] = ys - flow[:, x, 1]
        
        alpha = args.optical_flow
        img1morphedstyled = cv2.remap(img1styled, mapping, None, cv2.INTER_CUBIC)
        img2blend = blend(img2color, img1morphedstyled, alpha)
        image = Image.fromarray(np.uint8(img2blend[:, :, ::-1]))
    w, h = (int(args.scale*i) for i in image.size)
    orig = image.resize((w, h), 3)
    image = xp.asarray(orig, dtype=xp.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    x = Variable(image)

    y = None
    y = model(x)
    result = cuda.to_cpu(y.data)

    result = result.transpose(0, 2, 3, 1)
    result = result.reshape((result.shape[1:]))
    result = np.uint8(result)
    print filename, time.time() - start, 'sec'

    image = Image.fromarray(result)
    if args.ocol: image = original_colors(orig.resize((image.size), 3), image)
    image.save('{}{}'.format(args.out, os.path.basename(filename) if multi else ''))
