import time
import inspect
import atexit
import pickle
import os
import hashlib
import util

use_previous_session_information = False # Used to add percentage information to tprint output

_original_caller = inspect.stack()[-1].filename
_tprint_time_history = []
_tprint_previous_time_history_idx = -1
_tprint_previous_time_history = None
_config_folder = os.path.join(util.root(), "_tm_config")

if use_previous_session_information and not os.path.exists(_config_folder):
    os.makedirs(_config_folder)

def _init_previous_session(caller_id):
    global _tprint_previous_time_history, _tprint_previous_time_history_idx
    filename = f"sess_{str(hashlib.md5(caller_id.encode()).hexdigest())}.time"
    if not os.path.exists(os.path.join(_config_folder, filename)):
        return
    with open(os.path.join(_config_folder, filename), "rb") as f:
        arr = pickle.load(f)
    if len(arr) != 2:
        return
    if arr[0] != caller_id:
        return
    _tprint_previous_time_history = arr[1]
    _tprint_previous_time_history_idx = 0

if use_previous_session_information:
    _init_previous_session(_original_caller)

_last_time = time.time()
_overall_start_time = _last_time


def tprint(label=None):
    global _last_time, _tprint_time_history, _tprint_previous_time_history_idx
    current_time = time.time()
    print_string = f"{current_time - _last_time:>7.3f}s -"
    if not label is None:
        print_string += f" {label} -"
    history_prefix = None
    if _tprint_previous_time_history_idx >= 0 and _tprint_previous_time_history_idx < len(_tprint_previous_time_history):
        history_elem = _tprint_previous_time_history[_tprint_previous_time_history_idx]
        if label == history_elem[0]:
            percentage = 100 * history_elem[1]
            history_prefix = f"[{round(percentage):>3d}%]"
            _tprint_previous_time_history_idx += 1
        else:
            _tprint_previous_time_history_idx = -1
    if history_prefix is None:
        history_prefix = "-"
    print(f"{history_prefix} {print_string}")
    _tprint_time_history.append((label, current_time))
    _last_time = current_time

def elapsed_time():
    print(f"- Total runtime - {time.time() - _overall_start_time:>6.2f}s -")

def _exit_handler():
    total_time = time.time() - _overall_start_time
    if total_time <= 0 or not use_previous_session_information:
        return
    time_history = [(label, max(0, min(1, (ts - _overall_start_time) / total_time))) for label, ts in _tprint_time_history]
    to_save = [_original_caller, time_history]
    filename = f"sess_{str(hashlib.md5(_original_caller.encode()).hexdigest())}.time"
    # pickle
    with open(os.path.join(_config_folder, filename), "wb") as f:
        pickle.dump(to_save, f)
    
    elapsed_time()

atexit.register(_exit_handler)