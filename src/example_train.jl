# An example of how you can train this UNet and do inference
using Images
using FileIO
using JLD2

using Juno: @progress

using UNet

function check_img_size(img)
    # checks that image size is compatible with the unet
    height, width = size(img)
    if (height - 4) % 16 != 0
        throw(BoundsError())
    end
end

function load_img_label(number, crop=500)
    train_path = "data/membrane/train/"
    img_path = train_path*"image/"*string(number)*".png"
    label_path = train_path*"label/"*string(number)*".png"

    img = load(img_path)[1:crop, 1:crop]
    label = load(label_path)[1:crop, 1:crop]
    check_img_size(img)

    return img, label
end


# ---------------------------------------------
# Train a new model
# ---------------------------------------------

# build up dataset for training
dataset = []
for i = 0:20
    img, label = load_img_label(i)
    X = img2array(unet_tiling(img))
    y = img2array(label)
    push!(dataset, (X, y))
end

loss(x, y) = Flux.crossentropy(
    reshape(unet(x), size(y)[1]^2), reshape(y, size(y)[1]^2),
)
opt = ADAM(1e-3)

tx, ty = dataset[1]
evalcb = () -> @show loss(tx, ty)

@progress for i = 0:10
    @info "Epoch $i"
    Flux.train!(loss, params(unet), dataset, opt, cb = evalcb)
end

weights = params(unet)
@save "unet_weights.jld2" weights

# ---------------------------------------------
# And now, we can do some inference
# ---------------------------------------------

@load "unet_weights.jld2" weights
Flux.loadparams!(unet, weights)

img, label = load_img_label(4)
y_pred = unet(img2array(unet_tiling(img)))
array2img(y_pred)
