from __future__ import print_function
from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time
import riseml.server
import chainer
from chainer import cuda, Variable, serializers
from net import *
from io import BytesIO
import os


parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/seurat.model', type=str)
parser.add_argument('--median_filter', default=0, type=int)
parser.add_argument('--padding', default=50, type=int)

args = parser.parse_args()

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

def transfer_style(input_image):
	start = time.time()
	input_image = Image.open(BytesIO(input_image))
	image = np.asarray(input_image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
	image = image.reshape((1,) + image.shape)
	if args.padding > 0:
		image = np.pad(image, [[0, 0], [0, 0], [args.padding, args.padding], [args.padding, args.padding]], 'symmetric')
	image = xp.asarray(image)
	x = Variable(image)

	y = model(x)
	result = cuda.to_cpu(y.data)

	if args.padding > 0:
		result = result[:, :, args.padding:-args.padding, args.padding:-args.padding]
	result = np.uint8(result[0].transpose((1, 2, 0)))
	med = Image.fromarray(result)
	if args.median_filter > 0:
		med = med.filter(ImageFilter.MedianFilter(args.median_filter))
	print(time.time() - start, 'sec')
	output_image = BytesIO()
	med.save(output_image, format='JPEG')
	return output_image.getvalue()

riseml.server.serve(transfer_style, port=os.environ.get('PORT'))
