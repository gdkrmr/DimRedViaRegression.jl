
module DimRedViaRegression
export fit, fit!, predict, inverse, inverse!, DRR

import StatsBase
import StatsBase: fit, predict, predict!
import Iterators
# There is no inverse function in StatsBase

macro message(msg)
    return :(println(now(), ": ", $msg))
end

macro message(msg...)
    return :(println(now(), ": ", $msg...))
end


type DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}
    rotation ::Matrix{T}
    centers  ::Matrix{T}
    scales   ::Matrix{T}
    ndims    ::Int
    models   ::Vector{S}

    function DRR(rotation, scales, centers, ndims, models)
        @assert ndims > zero(ndims)
        @assert size(centers, 2)  == 1
        @assert size(scales, 2)   == 1
        @assert size(rotation, 1) == size(rotation, 2)
        @assert size(rotation, 1) == length(scales)
        @assert length(scales)    == length(centers)
        @assert ndims             == length(models) + 1
        new(rotation, centers, scales, ndims, models)
    end
end

function DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}(
    rotation ::Matrix{T},
    scales   ::Vector{T},
    centers  ::Vector{T},
    ndims    ::Int,
    models   ::Vector{S}
)
    DRR{T, S}(rotation, centers', scales', ndims, models)
end

function DRR{T <: AbstractFloat, S <: StatsBase.RegressionModel}(
    rotation ::Matrix{T},
    scales   ::Matrix{T},
    centers  ::Matrix{T},
    ndims    ::Int,
    models   ::Vector{S}
)
    DRR{T, S}(rotation, centers, scales, ndims, models)
end

"""
returns a DRR object and does not overwrite X

regpars is a tuple of arrays all combinations of array elements are tried by crossvalidation, if you do not want crossvalidation, set crossvalidate = 1 and regpars = ([par1], [par2])

for regpars = ([a, b], [c, d]) the combinations (a,c), (a,d), (b,c), (b,d) are tried.

internally reg = fit(regression, X, y, (a, c)...) and predict(reg, Xnew) is called and these functions must be defined.
"""
function fit{R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix,
    regression    ::Type{R},
    ndim          ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    XX = deepcopy(X)
    drr = fit!(DRR, XX, regression, ndim,
               rotate        = rotate,
               center        = center,
               scale         = scale,
               crossvalidate = crossvalidate,
               regpars       = regpars)
    return drr
end

"""
updates X with drr solution, returns the fitted DRR
"""
function fit!{R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix,
    regression    ::Type{R},
    ndim          ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    drr = fit_and_pca!(DRR, X, regression, ndim,
                       rotate        = rotate,
                       center        = center,
                       scale         = scale,
                       crossvalidate = crossvalidate,
                       regpars       = regpars)
    predict_no_rotate!(drr, X)
    return drr
end

"""
updates X with pca solution of X, returns the fitted DRR
"""
function fit_and_pca!{T, R <: StatsBase.RegressionModel}(
                  ::Type{DRR},
    X             ::Matrix{T},
    regression    ::Type{R},
    ndims         ::Int;
    rotate        ::Bool = true,
    center        ::Bool = true,
    scale         ::Bool = true,
    crossvalidate ::Int  = 10,
    regpars       ::Tuple = ()
)
    d, n = size(X)

    if center
        means = mean(X, 2)
        # substract_means!(X, means)
        broadcast!(-, X, X, means)
    else
        means = zeros(1, d)
    end

    if scale
        scales = std(X, 2)
        # scale_vals!(X, scales)
        broadcast!(/, X, X, scales)
    else
        scales = ones(T, d)
    end

    if rotate
        # Eigenvalue Decomposition:
        # AᵀA = VΣ²Vᵀ 
        # AAᵀ = UΣ²Uᵀ
        # SVD:
        # A  = UΣVᵀ
        # Aᵀ = VΣᵀUᵀ
        # SVD in julia is defined for row major order!!
        # eigenfact and svd give eigenvalues and vectors in different order!
        sv = svdfact(X, thin = false)
        ord = sortperm(sv.S; rev = true)
        rotation = sv.U[:, ord]

        X  = rotation' * X
        # This one does not work unfortunately:
        # BLAS.gemm!('T', 'N', one(T), rotation, X, zero(T), X)
    else
        rotation = eye(T, d)
    end

    models = Vector{regression}(ndims - 1)
    for i in d:-1:2
        if crossvalidate > 1
            models[i - 1] = crossvalidate_parameters(
                regression, X[1:d - 1, :], X[d, :], crossvalidate, regpars...
            )
        else
            models[i - 1] = fit(regression, X[1:d - 1, :], X[d, :], regpars...)
        end
    end
    # X[1,:] stays the same
    DRR(rotation, means, scales, ndims, models)
end

function inverse!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == drr.ndims

    inverse_no_rotate!(drr, Y)
    # BLAS.gemm!('N', 'N', one(T), rotation, Y, zero(T), Y)
    Y = drr.rotation * Y
    # Y = Y .+ drr.means
    broadcast!(+, Y, Y, drr.means)
    # Y = Y .* drr.scales
    broadcast!(*, Y, Y, drr.scales)
    return Y
end

function inverse{T}(drr::DRR{T}, Y::Matrix{T})
    YY = deepcopy(T)
    inverse!(drr, YY)
    return YY
end

function inverse_no_rotate!{T}(drr::DRR{T}, Y::Matrix{T})
    d, n = size(Y)
    @assert d == drr.ndims

    for i in d:-1:2
        # Y[d, :] += predict(drr.models[i - 1], Y[1:d - 1, :])
        # TODO: Make this inplace
        prd = predict(drr.models[i - 1], Y[1:d - 1, :])
        for j in 1:n
            Y[d, j] += prd[j]
        end
    end
    return Y
end

function predict!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)

    broadcast!(-, X, X, drr.centers)
    broadcast!(/, X, X, drr.scales)
    X = drr.rotation' * X
    # BLAS.gemm!('N', 'N', one(T), drr.rotation, X, zero(T), X)
    predict_no_rotate!(drr, X)

    return X
end

function predict{T}(drr::DRR{T}, X::Matrix{T})
    XX = deepcopy(X)
    predict!(drr, XX)
    return XX
end

function predict_no_rotate!{T}(drr::DRR{T}, X::Matrix{T})
    d, n = size(X)
    @assert d == drr.ndims

    for i in d:-1:2
        # X[d, :] = X[d, :] - predict(drr.models[i - 1], X[1:d - 1, :])
        # TODO: make this inplace
        prd = predict(drr.models[i - 1], X[1:d - 1, :])
        for j in 1:n
            @inbounds X[d, j] = prd[j]
        end
    end
    return X
end

function crossvalidate_parameters{T, S <: StatsBase.RegressionModel}(
    ::Type{S}, x::Matrix{T}, y::Vector{T}, folds::Int, pars...
)
    combs = Iterators.product(pars...)

    lossₘᵢₙ = typemax(T)
    mₘᵢₙ    = S
    combₘᵢₙ = Iterators.nth(combs, 1)

    n     = size(x)[end]
    xdim1 = size(x)[1:end - 1]
    ydim1 = size(y)[1:end - 1]
    perm_idxs    = shuffle(1:n)
    block_sizes  = make_blocks(n, folds)
    block_ends   = cumsum(block_sizes)
    block_starts = [1, (block_ends[1:end-1] + 1)... ]

    for comb in combs
        @message "Current:"
        @show comb

        loss = 0.0
        m = S
        for i in 1:folds
            print("$i ")
            idxsₜₑₛₜ = falses(n)
            idxsₜₑₛₜ[block_starts[i]:block_ends[i]] = true
            idxsₜᵣₐᵢₙ = ~idxsₜₑₛₜ

            x₂     = reshape(x, (prod(xdim1), size(x)[end]))
            y₂     = reshape(y, (prod(ydim1), size(y)[end]))
            xₜₑₛₜ  = reshape(x₂[:, idxsₜₑₛₜ], (xdim1..., block_sizes[i]))
            yₜₑₛₜ  = reshape(y₂[:, idxsₜₑₛₜ], (ydim1..., block_sizes[i]))
            xₜᵣₐᵢₙ = reshape(x₂[:, idxsₜᵣₐᵢₙ], (xdim1..., n - block_sizes[i]))
            yₜᵣₐᵢₙ = reshape(y₂[:, idxsₜᵣₐᵢₙ], (ydim1..., n - block_sizes[i]))

            m = fit(S, xₜᵣₐᵢₙ, yₜᵣₐᵢₙ, comb...)
            ŷ = predict(m, xₜₑₛₜ)
            loss += mean((yₜₑₛₜ - ŷ) .^ 2)
        end
        loss /= folds

        if loss < lossₘᵢₙ
            lossₘᵢₙ = loss
            mₘᵢₙ    = m
            combₘᵢₙ = comb
        end
        print("\n")
        @show loss
    end
    @message "Minimum:"
    @show combₘᵢₙ
    @show lossₘᵢₙ
    return mₘᵢₙ
end

function make_blocks(nobs, nblocks)
    maxbs, rest = divrem(nobs, nblocks)

    res = fill(maxbs, nblocks)
    if rest > 0
        res[1:rest] = maxbs + 1
    end
    res
end

# TODO: Make the following functions nicer!
# function add_means!{T <: AbstractFloat}(X::Matrix{T}, means::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds X[i, j] += means[i]
#         end
#     end
# end

# function substract_means!{T <: AbstractFloat}(X::Matrix{T}, means::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds X[i, j] -= means[i]
#         end
#     end
# end

# function unscale_vals!{T >: AbstractFloat}(X::Matrix{T}, scales::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds XX[i, j] *= scales[i]
#         end
#     end
# end

# function scale_vals!{T >: AbstractFloat}(X::Matrix{T}, scales::Vector{T})
#     d, n = size(X)
#     for j in 1:n
#         for i in 1:d
#             @inbounds XX[i, j] /= scales[i]
#         end
#     end
# end

end # module DRR
