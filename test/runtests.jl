using UNet
using Test

@info "Testing UNet architecture runs..."
@testset "UNet.jl" begin
    X = rand(256, 256, 1, 1)
    #
    # y = unet(X)

    # img[1:256, 1:256, :, :], label[95:162, 95:162, :, :]


    # Write your own tests here.
end
