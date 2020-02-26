module UNet

using Flux
using Flux: @treelike

struct UpSample
  ratio
end

@treelike UpSample

function (up::UpSample)(input)
  ratio = (up.ratio[1], up.ratio[2], 1, 1)
  (h, w, c, n) = size(input)
  y = similar(input, (ratio[1], 1, ratio[2], 1, 1, 1))
  fill!(y, 1)
  z = reshape(input, (1, h, 1, w, c, n))  .* y
  return reshape(z, size(input) .* ratio)
end

model = Chain(
    UpSample((1,1))
)

img = reshape(
    [1 1 1 1 0 0 0 0 0 0 0 0], 2, 2, 3, 1
)

conv_img = model(img)



drop4 = Chain(
    Conv((3,3), 3=>64, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3,3), 3=>64, relu, pad=(1, 1), stride=(1, 1)),
    MaxPool((2,2)),

    Conv((3,3), 64=>128, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3,3), 64=>128, relu, pad=(1, 1), stride=(1, 1)),
    MaxPool((2,2)),

    Conv((3,3), 128=>256, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3,3), 128=>256, relu, pad=(1, 1), stride=(1, 1)),
    MaxPool((2,2)),

    Conv((3,3),256=>512, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3,3), 256=>512, relu, pad=(1, 1), stride=(1, 1)),
    Dropout(0.5),
)

up6 = Chain(
    MaxPool((2,2)),
    Conv((3,3),512=>1024, relu, pad=(1, 1), stride=(1, 1)),
    Conv((3,3), 1024=>1024, relu, pad=(1, 1), stride=(1, 1)),
    Dropout(0.5),
    UpSample((2,2))
    Conv((2,2), 1024=>512)
)

unet = Chain(
    drop4,
    SkipConnection(up6, +)

)

end # module
