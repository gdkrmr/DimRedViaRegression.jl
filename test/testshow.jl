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
show(helix_drr)
