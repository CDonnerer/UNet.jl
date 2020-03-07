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


function get_Xy_train(number)
    img = load_img_array(
        "data/membrane/train/image/"*string(number)*".png"
    )
    label = load_img_array(
        "data/membrane/train/label/"*string(number)*".png"
    )
    # img[1:256, 1:256, :, :], label[95:162, 95:162, :, :]
    return img[1:512, 1:512, :, :], label[95:418, 95:418, :, :]
end

for i = 0:15
    X, y = get_Xy_train(i)
    push!(dataset, (X,y))
end

y_pred = unet(dataset[1][1])

loss(x, y) = Flux.crossentropy(
    reshape(unet(x), size(y)[1]^2), reshape(y, size(y)[1]^2),
)

opt = ADAM(0.0003)

using Flux: throttle, params, crossentropy
using Juno: @progress

tx, ty = dataset[5]
evalcb = () -> @show loss(tx, ty)

@progress for i = 1:20
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


X, y = get_Xy_train(18)
y_pred = unet(X)
plot_y(X)
plot_y(y)
plot_y(y_pred)
