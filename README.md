# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

- input image size: 1024x768
- process time(CPU): 17.78sec (Core i7-5930K)
- process time(GPU): 0.994sec (GPU TitanX)

## Differences from original
#### Training
* default `--image_size` set to 512 (original uses 256). It's slow, but time is the price you have to pay for quality
* ability to switch off dataset cropping with `--fullsize` option. Crops by default to preserve aspect ratio
* cropping implementation uses [`ImageOps.fit`](http://pillow.readthedocs.io/en/3.1.x/reference/ImageOps.html#PIL.ImageOps.fit), which always scales and crops, whereas original uses custom solution, which upscales the image if it's smaller than `--image_size`, otherwise just crops without scaling
* bicubic and Lanczos [resampling](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize) when scaling dataset and input style images respectively provides sharper shrinking, whereas original uses nearest neighbour

#### Generating
* Ability to specify multiple files for input to exclude model reloading every iteration. The format is standard Unix path expansion rules, like `file*` or `file?.png` Don't forget to quote, otherwise the shell will expand it first. Saves about 0.5 sec on each image.
* Output specifies path prefix if multiple files are used for input, otherwise an explicit filename
* Option `-x` indicates content image scaling factor before transformation
* Preserve original content colors with `--original_colors` flag. More info: [Transfer style but not the colors](https://github.com/jcjohnson/neural-style/issues/244)

## Video Processing
The repo includes a bash script to transform your videos. It depends on ffmpeg. [Compilation instructions](https://trac.ffmpeg.org/wiki/CompilationGuide)
```
./genvid.sh input_video output_video model start_time duration
```
The first three arguments are mandatory and should contain path to files.<br>
The last two are optional and indicate starting position and duration in seconds.

## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolutional layers which are 10% of the entire model.
```
sh setup_model.sh
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py -s <style_image_path> -d <training_dataset_path> -g 0
```

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path>
```

This repo has pretrained models as an example.

- example:
```
python generate.py sample_images/tubingen.jpg -m models/composition.model -o sample_images/output.jpg
```
or
```
python generate.py sample_images/tubingen.jpg -m models/seurat.model -o sample_images/output.jpg
```

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) causes unstable result.

## No Backward Compatibility
##### Jul. 19, 2016
This version is not compatible with the previous versions. You can't use models trained by the previous implementation. Sorry for the inconvenience!

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) Chainer implementation of neural-style. I heavily referenced it.
- [chainer-cifar10](https://github.com/mitmul/chainer-cifar10) Residual block implementation is referred.
