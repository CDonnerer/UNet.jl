
img2array(img) = reshape(
    convert(Array{Float32}, img), size(img) ..., 1, 1
)

array2img(x) = convert(
    Array{Gray{Normed{UInt8,8}},2},
    reshape(x.data, size(x)[1], size(x)[2])
)

function unet_dimensions(input)
    output = input - 4
    for depth in 1:4
        output = output ÷ 2 - 4
    end

    for expand in 1:4
        output = 2 * output -4
    end

    return output
end

function required_padding(img)
    # calculate how much padding is needed around img
    height, width = size(img)
    output_height = unet_dimensions(height)

    return (height - output_height) ÷ 2
end

function unet_tiling(img)
    # given an input image, tiles it for the unet
    padding = required_padding(img)
    return padarray(img, Pad(:symmetric, padding, padding))
end
