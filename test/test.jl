
using RDatasets
import DimRedViaRegression
import KernelRidgeRegression
import MLKernels
using Gallium
using Plots
pyplot()

iris = dataset("datasets", "iris")
x = convert(Matrix, iris[1:4])'


iris_drr = fit(DimRedViaRegression.DRR, x, KernelRidgeRegression.KRR, 4,
               rotate = true, center = true, scale = true, crossvalidate = 10,
               regpars = (logspace(-5, 2, 8),
                          [MLKernels.GaussianKernel(x)
                           for x in logspace(-5, 2, 8)]))
y = predict(iris_drr, x)

# show(y)

# X = deepcopy(x)
# T = eltype(X)
# Y = rotation' * X

# XX = deepcopy(X)
# BLAS.gemm!('T', 'N', one(T), rotation, XX, one(T), XX)

# Y - XX

# DimRedViaRegression.fit_and_pca!(DimRedViaRegression.DRR, xx, KernelRidgeRegression.KRR, 4,
#                                  rotate = true, center = true, scale = true, crossvalidate = 10,
#                                  regpars = (logspace(-5, 2, 8),
#                                             [MLKernels.GaussianKernel(x)
#                                              for x in logspace(-5, 2, 8)]))



scatter(x[1,:], x[2,:], group = iris[5])
scatter(y[1,:], y[2,:], group = iris[5])

using Gadfly

Gadfly.plot(x = x[1,:], y = x[2,:], group = iris[5])
Gadfly.plot(x = y[1,:], y = y[2,:], group = iris[5])
