ROOT_FOLDER = ""

import os
import numpy as np

if not os.path.exists(os.path.join(ROOT_FOLDER, "run_config.json")):
    raise Exception(f"No run_config found in root folder {ROOT_FOLDER}. Please specify the correct path in the first line of util.py")

def root():
    return ROOT_FOLDER

def prettify(l):
    return f'{np.mean(l):.5f} +- {np.std(l):.5f}  (min: {np.min(l)})'

## --- Exceptions ---

class ArchitectureNotCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture needs to be compiled to perform this operation")


class ArchitectureCompiledException(Exception):
    def __init__(self):
        super().__init__("Architecture is already compiled, cannot perform this operation")
