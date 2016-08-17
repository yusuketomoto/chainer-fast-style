import numpy as np
import argparse
from PIL import Image
import time, os, glob

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
parser.add_argument('--original_colors', dest='ocol', action='store_true')
parser.set_defaults(ocol=False)
args = parser.parse_args()

def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

imagelist = glob.glob(args.input)
multi = True if len(imagelist) > 1 else False
for filename in imagelist:
    start = time.time()
    image = Image.open(filename).convert('RGB')
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

    result = Image.fromarray(result)
    if args.ocol: result = original_colors(orig.resize((result.size), 3), result)
    result.save('{}{}'.format(args.out, os.path.basename(filename) if multi else ''))
