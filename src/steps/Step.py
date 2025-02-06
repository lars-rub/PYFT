from src import util_jax

class Step:
    def __init__(self, name, params):
        self._name = name
        self._params = params
        self._max_incoming_connections = 1
        self.is_dynamic = False
        self.needs_input_connections = True
        self.is_source = False
        self._buf = {}

    def get_max_incoming_connections(self):
        return self._max_incoming_connections

    def get_name(self):
        return self._name
    
    def reset(self):
        self._output_buf = None

    def set_output_buffer(self, output):
        self._output_buf = output

    def get_output_buffer(self):
        return self._output_buf
    
    def get_buffer(self, buf_name):
        return self._buf[buf_name]
    
    def update_input(self, arch):
        input_sum = None
        incoming_steps = arch.get_incoming_steps(self.get_name())
        if len(incoming_steps) == 0:
            if self.needs_input_connections:
                raise ValueError(f"Step {self.get_name()} has no incoming connection")
            input_sum = util_jax.zeros(self._params["shape"], device_idx=self._device_idx)
        else:
            for step in incoming_steps:
                result = step.get_output_buffer()
                if input_sum is None:
                    input_sum = result
                else:
                    input_sum += result
        return input_sum
    
