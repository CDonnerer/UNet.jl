# Example script of how you can train this UNet and do inference
using Images, FileIO, JLD2, Flux, UNet

using Juno: @progress


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
    img, label = load_img_label(i, 340)
    X = img2array(unet_tiling(img))
    y = img2array(label)
    push!(dataset, (X, y))
end

loss(x, y) = Flux.crossentropy(
    reshape(unet(x), size(y)[1]^2), reshape(y, size(y)[1]^2),
)
opt = Flux.ADAM(1e-4)

# get a validation datapoint
img, label = load_img_label(25, 340)
X_valid = img2array(unet_tiling(img))
y_valid = img2array(label)

evalcb = () -> @show loss(X_valid, y_valid)

@progress for i = 0:5
    @info "Epoch $i"
    Flux.train!(loss, params(unet), dataset, opt, cb = evalcb)
end

weights = params(unet)
@save "unet_weights.jld2" weights

# ---------------------------------------------
# And now, we can do some inference at last
# ---------------------------------------------

@load "unet_weights.jld2" weights
Flux.loadparams!(unet, weights)

img, label = load_img_label(21)
y_pred = unet(img2array(unet_tiling(img)))

img_pred = array2img(y_pred)
