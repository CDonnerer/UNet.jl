using UNet
using Test

@info "Testing UNet architecture runs..."
@testset "UNet.jl" begin
    X = rand(256, 256, 1, 1)
    y = unet(X)
    @test size(y) == (68, 68, 1, 1)
end
