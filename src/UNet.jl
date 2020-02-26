module UNet

using Flux

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
