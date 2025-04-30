from src import util_jax
from src import util
from src.Configurable import Configurable

class Slot():
    def __init__(self, step, slot_name):
        self._step = step
        self.name = step.get_name() + "." + slot_name

    def __rshift__(self, other):
        if isinstance(other, Step):
            util.get_architecture().connect_to(self.name, other.get_name())
        elif isinstance(other, str):
            util.get_architecture().connect_to(self.name, other)
        elif isinstance(other, Slot):
            util.get_architecture().connect_to(self.name, other.name)
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other):
        if isinstance(other, Step):
            util.get_architecture().connect_to(other.get_name(), self.name)
        elif isinstance(other, str):
            util.get_architecture().connect_to(other, self.name)
        elif isinstance(other, Slot):
            util.get_architecture().connect_to(other.name, self.name)
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

class Step(Configurable):
    def __init__(self, name, params, mandatory_params, is_dynamic=False):
        super().__init__(params, mandatory_params)
        self._name = name
        if "." in name:
            raise ValueError(f"Step names cannot contain dots. ({name})")
        self.is_dynamic = is_dynamic
        if is_dynamic and not "shape" in params:
            raise ValueError(f"Dynamic steps require a shape parameter. ({name})")
        self._max_incoming_connections = {}
        self.needs_input_connections = True
        self.is_source = False
        self.buffer = {} # Stores matrices of internal and output buffers
        self.output_slot_names = []
        self.input_slot_names = [] 
        self.register_input(util.DEFAULT_INPUT_SLOT)
        self.register_output(util.DEFAULT_OUTPUT_SLOT)

    def compute(self, input_mats, **kwargs):
        raise NotImplementedError("Please override compute() in subclasses of Step.")

    def get_max_incoming_connections(self, slot_name):
        return self._max_incoming_connections[slot_name]

    def get_name(self): # TODO remove get methods?
        return self._name
    
    def reset(self):
        for name in self.output_slot_names:
            self.reset_buffer(name)
        
    def reset_buffer(self, slot_name):
        if not self.is_dynamic:
            self.buffer[slot_name] = None
        else:
            self.buffer[slot_name] = util_jax.zeros(self._params["shape"])

    def pre_compile(self, arch):
        if not self.is_source:
            incoming_steps = arch.get_incoming_steps(self.get_name())
            if len(incoming_steps) == 0 and self.needs_input_connections:
                raise ValueError(f"Step {self.get_name()} has no incoming connection")
    
    def get_buffer(self, buf_name):
        if buf_name in self.buffer:
            return self.buffer[buf_name]
        else:
            raise Exception(f"Buffer {buf_name} not found ({self.get_name()})")
        
    def __rshift__(self, other):
        if isinstance(other, Step):
            util.get_architecture().connect_to(self.get_name(), other.get_name())
        elif isinstance(other, str):
            util.get_architecture().connect_to(self.get_name(), other)
        elif isinstance(other, Slot):
            util.get_architecture().connect_to(self.get_name(), other.name)
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")

    def __lshift__(self, other):
        if isinstance(other, Step):
            util.get_architecture().connect_to(other.get_name(), self.get_name())
        elif isinstance(other, str):
            util.get_architecture().connect_to(other, self.get_name())
        elif isinstance(other, Slot):
            util.get_architecture().connect_to(other.name, self.get_name())
        else:
            raise Exception(f"Cannot connect to unknown type ({type(other)})")
    
    def register_output(self, slot_name):
        # Initialize output buffer
        self.register_buffer(slot_name)
        # Register output slot shortcut
        setattr(self, f"o{len(self.output_slot_names)}", Slot(self, slot_name))
        # Save output slot name
        if slot_name in self.output_slot_names:
            raise ValueError(f"Output slot {slot_name} already registered in step {self.get_name()}")
        self.output_slot_names.append(slot_name)

    def register_input(self, slot_name, max_incoming_connections=1):
        # Register input slot shortcut
        setattr(self, f"i{len(self.output_slot_names)}", Slot(self, slot_name))
        # Register input slot
        if slot_name in self.input_slot_names:
            raise ValueError(f"Input slot {slot_name} already registered in step {self.get_name()}")
        self.input_slot_names.append(slot_name)
        # Save max incoming connections
        self._max_incoming_connections[slot_name] = max_incoming_connections
    
    def register_buffer(self, buf_name):
        if buf_name in self.buffer:
            raise ValueError(f"Buffer {buf_name} already registered ({self.get_name()})")
        self.reset_buffer(buf_name)

    def update_input(self, arch):
        input_sums = {}
        for input_slot in self.input_slot_names:
            input_sum = None
            incoming_steps = arch.get_incoming_steps(self.get_name() + "." + input_slot)
            if len(incoming_steps) == 0:
                input_sum = util_jax.zeros(self._params["shape"])
            else:
                for step_slot in incoming_steps:
                    step, slot = step_slot.split(".")
                    # Get output buffer of connected step and add it to the input sum
                    step_output = arch.get_element(step).get_buffer(slot)
                    input_sum = input_sum + step_output if input_sum is not None else step_output
            if input_sum is None:
                raise ValueError(f"Step {self.get_name()} has no valid input sum at slot {input_slot}. This should never happen")
            input_sums[input_slot] = input_sum
        return input_sums
    
    # Only gets called in dynamic steps
    def post_compute(self, output_matrices):
        # Output of compute is now ready, save it to output buffer
        for key in output_matrices.keys():
            output = output_matrices[key]
            output.block_until_ready()
            self.buffer[key] = output
