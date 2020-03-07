# UNet

UNet implementation for image segmentation in Julia build on top of the [Flux](http://fluxml.github.io/) library.

## Eample usage

```julia
using UNet

# Create dummy image data (single gray channel)
X = rand(Float32, 224, 224, 1, 1)
256×256×1×1 Array{Float32,4}:

y = unet(X)
68×68×1×1 Array{Float32,4}
```

## Notes on architecture

Please note that this UNet uses the implementation descibed in the original paper [Ronneberger et al.](https://arxiv.org/abs/1505.04597).

In this implementation, the UNet uses convolutional layers without padding, which results in the output prediction is smaller than the input. The output predicted segmentation will hence correspond to only the central area of the input, and a uniform border of 94 pixels around it will not be predicted. In order to obtain a segmentation prediction for the full input image, *tiling* has to be used.
