import util

class Step:
    def __init__(self, name, params):
        self._name = name
        self._params = params
        self._max_incoming_connections = 1
        self.is_dynamic = False


    def get_max_incoming_connections(self):
        return self._max_incoming_connections

    def get_name(self):
        return self._name
    
    def reset(self):
        self._input = None
        self._output_buf = None # util.zeros(self._params["shape"], dtype=util.cfg["dtype"])

    def set_output_buffer(self, output):
        self._output_buf = output

    def get_output_buffer(self):
        return self._output_buf
    
