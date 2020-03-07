module UNet

using Flux
using Flux: @treelike

export unet

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

function crop(original, desired)
    desired_h, desired_w = size(desired)
    original_h, original_w = size(original)

    delta_h = convert(Int8, (original_h - desired_h) ÷ 2)
    delta_w = convert(Int8, (original_w - desired_w) ÷ 2)

    return original[
        1+delta_h:delta_h+desired_h,
        1+delta_w:delta_w+desired_w,
        :, :]
end

struct CroppedSkipConcat
    layers
    concat_axis
end

@treelike CroppedSkipConcat

function (skip::CroppedSkipConcat)(input)
    output = skip.layers(input)
    cropped_input = crop(input, output)
    return cat(output, cropped_input, dims=3)
end


depth_4 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 512=>1024, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 1024=>1024, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 1024=>512, pad=(0,0), stride=(1,1)),
)

depth_3 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 256=>512, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 512=>512, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConcat(depth_4, +),
    Conv((3,3), 1024=>512, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 512=>512, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 512=>256, pad=(0,0), stride=(1,1)),
)

depth_2 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 128=>256, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 256=>256, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConcat(depth_3, +),
    Conv((3,3), 512=>256, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 256=>256, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 256=>128, pad=(0,0), stride=(1,1)),
)

depth_1 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 64=>128, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 128=>128, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConcat(depth_2, +),
    Conv((3,3), 256=>128, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 128=>128, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 128=>64, pad=(0,0), stride=(1,1)),
)

unet = Chain(
    Conv((3,3), 1=>64, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConcat(depth_1, +),

    Conv((3,3), 128=>64, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(0,0), stride=(1,1)),

    Conv((1,1), 64=>1, pad=(0,0), stride=(1,1)),

    # Finally, normalise and softmax to get nice probabilities
    BatchNorm(1),
    x -> sigmoid.(x)
)

end # module
