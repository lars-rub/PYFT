
# Used for parameterizable objects such as steps or kernels.
class Configurable:

    def __init__(self, params, mandatory_params):
        self._params = params
        for param in mandatory_params:
            if not param in params:
                raise ValueError(f"Parameter {param} is mandatory.")