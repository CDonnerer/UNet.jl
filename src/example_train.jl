# An example of how you can train this UNet and do inference

using Images
using FileIO

using Flux: throttle, params, crossentropy
using Juno: @progress


function load_img_array(path)
    img = load(path)
    reshape(convert(Array{Float32}, img), size(img) ..., 1, 1)
end

function get_Xy_train(number)
    train_path = "data/membrane/train/"

    img = load_img_array(train_path*"image/"*string(number)*".png")
    label = load_img_array(train_path*"label/"*string(number)*".png")
    return img[1:256, 1:256, :, :], label[95:162, 95:162, :, :]
    # use the below for larger size of input/ output
    #return img[1:512, 1:512, :, :], label[95:418, 95:418, :, :]
end


dataset = []
for i = 0:2
    X, y = get_Xy_train(i)
    push!(dataset, (X,y))
end

X, y = get_Xy_train(18)

X = rand(256, 256, 1, 1)
y = rand(68, 68, 1, 1)
dataset = [(X, y)]

loss(x, y) = Flux.crossentropy(
    reshape(unet(x), size(y)[1]^2), reshape(y, size(y)[1]^2),
)

y_pred = unet(X)

one = reshape(unet(X), size(y)[1]^2)
two = reshape(y, size(y)[1]^2)

Flux.crossentropy(one, two)

loss(X, y)

opt = ADAM(0.0003)

tx, ty = dataset[1]
evalcb = () -> @show loss(tx, ty)

@progress for i = 0:1
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

using JLD2
@load "unet_weights.jld2" weights

untracked_weights = data.(weights.params)

using Flux
Flux.loadparams!(unet, untracked_weights)



X, y = get_Xy_train(18)
y_pred = unet(X)
plot_y(X)
plot_y(y)
plot_y(y_pred)
