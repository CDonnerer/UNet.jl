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

struct CroppedSkipConnection
    layers
    connection
end

@treelike CroppedSkipConnection

function (skip::CroppedSkipConnection)(input)
    output = skip.layers(input)
    cropped_input = crop(input, output)
    return cat(output, cropped_input, dims=3)
end


depth_4 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 512=>1024, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 1024=>1024, relu, pad=(0,0), stride=(1,1)),
    #Dropout(0.5),
    UpSample((2,2)),
    Conv((1,1), 1024=>512, pad=(0,0), stride=(1,1)),
)

depth_3 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 256=>512, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 512=>512, relu, pad=(0,0), stride=(1,1)),
    #Dropout(0.5),

    CroppedSkipConnection(depth_4, +),
    Conv((3,3), 1024=>512, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 512=>512, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 512=>256, pad=(0,0), stride=(1,1)),
)

depth_2 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 128=>256, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 256=>256, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConnection(depth_3, +),
    Conv((3,3), 512=>256, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 256=>256, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 256=>128, pad=(0,0), stride=(1,1)),
)

depth_1 = Chain(
    MaxPool((2,2)),
    Conv((3,3), 64=>128, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 128=>128, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConnection(depth_2, +),
    Conv((3,3), 256=>128, relu, pad=(0,0), stride=(1,1)),

    Conv((3,3), 128=>128, relu, pad=(0,0), stride=(1,1)),
    UpSample((2,2)),
    Conv((1,1), 128=>64, pad=(0,0), stride=(1,1)),
)

unet = Chain(
    Conv((3,3), 1=>64, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(0,0), stride=(1,1)),

    CroppedSkipConnection(depth_1, +),

    Conv((3,3), 128=>64, relu, pad=(0,0), stride=(1,1)),
    Conv((3,3), 64=>64, relu, pad=(0,0), stride=(1,1)),

    Conv((1,1), 64=>1, pad=(0,0), stride=(1,1)),
    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    #x -> reshape(x, :, size(x, 4)),

    # Finally, softmax to get nice probabilities
    BatchNorm(1),
    x -> sigmoid.(x)
)

# img = rand(256, 256, 1, 1)
# seg = unet(img)

using Images
using FileIO


function load_img_array(path)
    img = load(path)
    reshape(convert(Array{Float32}, img), size(img) ..., 1, 1)
end

img = load_img_array("data/membrane/train/image/0.png")
label = load_img_array("data/membrane/train/label/0.png")

# 256 input
# 68 label

X_train = img[1:256, 1:256, :, :]
y_train = label[95:162, 95:162, :, :]

a = []

push!(a, (X_train, y_train))

dataset = []

for i = 0:10
    X, y = get_Xy_train(i)
    push!(dataset, (X,y))
end

y_pred = unet(X_train)

loss(x, y) = Flux.crossentropy(
    reshape(unet(x), size(y)[1]^2), reshape(y, size(y)[1]^2),
)

opt = ADAM(0.00001 d)

using Flux: throttle, params, crossentropy
using Juno: @progress

tx, ty = dataset[2]
evalcb = () -> @show loss(tx, ty)

@progress for i = 1:2
    @info "Epoch $i"
    Flux.train!(loss, params(unet), dataset, opt, cb = evalcb)
end

function plot_y(y)
    convert(
        Array{Gray{Normed{UInt8,8}},2},
        reshape(y.data, size(y)[1], size(y)[2])
    )
end

function plot_y(y::Array{Float32,4})
    convert(
        Array{Gray{Normed{UInt8,8}},2},
        reshape(y, size(y)[1], size(y)[2])
    )
end


function get_Xy_train(number)
    img = load_img_array(
        "data/membrane/train/image/"*string(number)*".png"
    )
    label = load_img_array(
        "data/membrane/train/label/"*string(number)*".png"
    )
    return img[1:256, 1:256, :, :], label[95:162, 95:162, :, :]
end

y_pred = unet(X_train)
plot_y(y_pred)
plot_y(y_train)
plot_y(img[95:162, 95:162, :, :])
plot_y(y_train)

convert(
    Array{Gray{Normed{UInt8,8}},2},
    reshape(y_pred.data, 68, 68)
)

X, y = get_X_y(1)
y_pred = unet(X)
plot_y(X)
plot_y(y)
plot_y(y_pred)

end # module
