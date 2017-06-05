
import DimRedViaRegression
import KernelRidgeRegression
n = 500
tt = linspace(0, 4π, n)
helix = hcat(
    3 * cos(tt) + (randn(n) .* linspace(0.1, 1.4, n)),
    3 * sin(tt) + (randn(n) .* linspace(0.1, 1.4, n)),
    3 * tt      + (randn(n) .* linspace(0.1, 1.4, n)),
)'
helix_drr = DimRedViaRegression.fit(DimRedViaRegression.DRR, helix,
                KernelRidgeRegression.KRR, 3,
                rotate = true, center = true, scale = false,
                crossvalidate = 4,
                regpars = (logspace(-5, 2, 8),
                           [MLKernels.GaussianKernel(x)
                            for x in logspace(-5, 2, 8)]))
helix_drr
showcompact(helix_drr)

(
    code_lowered(x->x^2)[1].
)

import StatsBase
import StatsBase.fit
import DimRedViaRegression
import KernelRidgeRegression
import MLKernels
using Plots
plotlyjs()

reload("DimRedViaRegression")

n = 500
tt = linspace(0, 4π, n)

helix = hcat(
    3 * cos(tt) + (randn(n) .* linspace(0.1, 1.4, n)),
    3 * sin(tt) + (randn(n) .* linspace(0.1, 1.4, n)),
    3 * tt      + (randn(n) .* linspace(0.1, 1.4, n)),
)'

scatter(helix[1,:], helix[2,:], helix[3,:])

helix_drr = DimRedViaRegression.fit(DimRedViaRegression.DRR, helix,
                KernelRidgeRegression.KRR, 3,
                rotate = true, center = true, scale = false,
                crossvalidate = 4,
                regpars = (logspace(-5, 2, 8),
                           [MLKernels.GaussianKernel(x)
                            for x in logspace(-5, 2, 8)]))
helix_drr
showcompact(helix_drr)

helix_drr_fit = DimRedViaRegression.predict(helix_drr, helix)
scatter(helix_drr_fit[1,:], helix_drr_fit[2,:], helix_drr_fit[3,:])

helix_recon = DimRedViaRegression.inverse(helix_drr, helix_drr_fit)
scatter(helix_recon[1,:], helix_recon[2,:], helix_recon[3,:])
helix_recon - helix

helix_backbone = DimRedViaRegression.inverse(
    helix_drr, hcat(helix_drr_fit[1, :],
                    linspace(0, 0, n) ,
                    linspace(0, 0, n))'
)
scatter(helix_backbone[1,:], helix_backbone[2,:], helix_backbone[3,:])

helix_2d_recon = DimRedViaRegression.inverse(
    helix_drr, hcat(helix_drr_fit[1:2, :]',
                    linspace(0, 0, n))'
)
scatter(helix_2d_recon[1,:], helix_2d_recon[2,:], helix_2d_recon[3,:])




reload("DimRedViaRegression")
helix_rotated = deepcopy(helix)
DimRedViaRegression.fit_and_pca!(DimRedViaRegression.DRR, helix_rotated,
                                 KernelRidgeRegression.KRR, 3, scale = false)
scatter(helix_rotated[1,:], helix_rotated[2,:], helix_rotated[3,:])



helix_man_fit = deepcopy(helix_rotated)
DimRedViaRegression.predict_no_rotate!(helix_drr, helix_man_fit)
scatter(helix_man_fit[1,:], helix_man_fit[2,:], helix_man_fit[3,:])


rotated_helix = helix_drr.rotation' * (helix .- helix_drr.centers)
scatter(rotated_helix[1, :], rotated_helix[2, :], rotated_helix[3, :])

helix_rotated_drr = DimRedViaRegression.DRR(eye(3), ones(3, 1), zeros(3, 1), 3, helix_drr.models)
rotated_helix_fit = DimRedViaRegression.predict(helix_rotated_drr, rotated_helix)

helix_rotated_backbone = DimRedViaRegression.inverse(

)

eye(3)' * ((rotated_helix .- zeros(3, 1)) ./ ones(3, 1))
predict_rotated_helix = DimRedViaRegression.predict_no_rotate!(helix_rotated_drr, deepcopy(rotated_helix))
scatter(predict_rotated_helix[1, :], predict_rotated_helix[2, :], predict_rotated_helix[3, :])

helix_drr_fit - predict_rotated_helix

recon_no_pca_3d = DimRedViaRegression.inverse_no_rotate!(helix_rotated_drr, deepcopy(predict_rotated_helix))
recon_no_pca_2d = DimRedViaRegression.inverse_no_rotate!(helix_rotated_drr,
                                                         hcat(predict_rotated_helix[1, :],
                                                              predict_rotated_helix[2, :],
                                                              zeros(500))')
recon_no_pca_1d = DimRedViaRegression.inverse_no_rotate!(helix_rotated_drr,
                                                         hcat(predict_rotated_helix[1, :], zeros(500), zeros(500))')

scatter(recon_no_pca_1d[1, :], recon_no_pca_1d[2, :], recon_no_pca_1d[3, :], )
scatter(recon_no_pca_2d[1, :], recon_no_pca_2d[2, :], recon_no_pca_2d[3, :], )
scatter(recon_no_pca_3d[1, :], recon_no_pca_3d[2, :], recon_no_pca_3d[3, :], )

sqrt.(mean((rotated_helix - recon_no_pca_1d) .^ 2, 1))
sqrt.(mean((helix - helix_backbone) .^ 2, 1))


X = helix
T = eltype(X)
d = size(X, 1)
ndims = d
regression = Type{KernelRidgeRegression.KRR}
crossvalidate = 4
regpars = (logspace(-5, 2, 8),
           [MLKernels.GaussianKernel(x)
            for x in logspace(-5, 2, 8)])
regpars = (1e-5, MLKernels.GaussianKernel(1e-3))
import DimRedViaRegression.crossvalidate_parameters

scatter(X[1,:], X[2,:], X[3,:])

MLKernels.kernelmatrix(MLKernels.ColumnMajor(),
                        MLKernels.GaussianKernel(1.0),
                        randn(3, 4), randn(3, 5))

