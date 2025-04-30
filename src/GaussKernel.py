import jax.numpy as jnp
import jax
from src import util_jax
from src.Configurable import Configurable

class GaussKernel(Configurable):

    def __init__(self, params):
        mandatory_params = ["sigma", "amplitude", "normalized"]
        super().__init__(params, mandatory_params)
        self._dimensionality = len(params["sigma"])
        # Estimate width if not explicitly set
        if not "shape" in params:
            self._params["shape"] = self._estimate_size()
        self._kernel = self.gkern_for_all_shapes(params["normalized"])
        if "max_shape" in params:
            self.check_kernel_size()
        if len(params["shape"]) != len(params["sigma"]):
            raise ValueError(f"GaussKernel requires equal dimensionality of sigma ({len(params['sigma'])}) and shape ({len(params['shape'])})")

    def _estimate_size(self):
        limit = 5
        widths = []
        for dim in range(self._dimensionality):
            sigma = self._params["sigma"][dim]
            if sigma == 0:
                widths.append(1)
            else:
                width = int(jnp.ceil(limit * sigma))
                if width % 2 == 0:
                    width += 1
                widths.append(width)
        return widths
    
    def check_kernel_size(self):
        kernel_size = self._params["shape"]
        max_shape = self._params["max_shape"]
        if len(kernel_size) != len(max_shape):
            raise Exception(f"Dimensionality of kernel size and max_shape must be equal ({len(kernel_size)} != {len(max_shape)})")

        for dim in range(len(kernel_size)):
            if kernel_size[dim] > max_shape[dim]:
                # Trim the kernel in the current dimension to be max_shape[dim] wide (or -1 if not odd)
                new_size = max_shape[dim] - (1 if max_shape[dim] % 2 == 0 else 0)
                #self._kernel = self._kernel[tuple(slice(None) if i != dim else slice(new_size) for i in range(len(kernel_size)))]
                # Trim it so that the center of the kernel is at the center of the max_shape
                center = kernel_size[dim] // 2
                new_center = new_size // 2
                start = center - new_center
                end = start + new_size
                slices = [slice(None) if i != dim else slice(start, end) for i in range(len(kernel_size))]
                self._kernel = self._kernel[tuple(slices)]
                self._params["shape"][dim] = new_size

    def gkern_for_all_shapes(self, normalize):
        # creates gaussian kernel with specified side lenghts and sigma
        shape = self._params["shape"]
        kernel = None
        for dim in range(len(shape)):
            ax = jnp.linspace(-(shape[dim] - 1) / 2., (shape[dim] - 1) / 2., shape[dim], dtype=util_jax.cfg["jdtype"])
            gauss_1d = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(self._params["sigma"][dim]))
            if normalize:
                gauss_1d /= jnp.sum(gauss_1d)

            if kernel is None:
                kernel = gauss_1d
            else:
                kernel = jnp.outer(kernel, gauss_1d).reshape(shape[:dim+1])
        return self._params["amplitude"] * kernel
    
    def get_kernel(self):
        return self._kernel
    