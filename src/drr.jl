
module DRR

import KRR

type DRR{T}
    rotation::StridedMatrix{T}
    scales::StridedVector{T}
    centers::StridedVector{T}
    ndims::Int
    models::Vector{KRR.AbstractKRR}
end


function fit{T}(drr::DRR{T}, X::StridedMatrix{T}, ndim::Int = 0;
                rotate = true, center = true, scale = true, )
    n, d = size(X)
    XX = deepcopy(X)

    if center
        means = mean(X, 1)
        for i in 1:d, j in 1:n
            XX[j, i] -= means[i]
        end
    else
        means = zeros(, d)
    end

    if scale
        scales = std(X, 1)
        for i in 1:d, j in 1:n
            XX[j, i] /= scales[i]
        end
    else
        scales = ones(T, d)
    end

    if rotate
        rotation = svd(XX)[:U]
        XX = rotation * XX
    else
        rotation = eye(T, d)
    end

    models = Vector{KRR.KRR, ndim}()
    for i in d:-1:2
        models[i] = KRR.KRR(XX[:, 1:(d-1)])
    end
    

end


end # module DRR
