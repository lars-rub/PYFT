#!/usr/bin/python3
import os
python_prefix = "python pynfit.py architectures/perf_0ND.py"

cmd = None
for i_a, arch_config in enumerate(["2 3 50", "2 30 50", "2 300 50", "2 30 500", "3 3 50", "4 3 50", "3 30 50", "4 30 50"]):
    for cpu in ["", "--cpu"]:
        if i_a > 4 and cpu == "--cpu":
            continue
        for comp_type in ["", "--use_vmap", "--eulerstep_static_args"]:
            args = " --arch_args " + arch_config + " --save_timing " + cpu + " " + comp_type
            if cmd is None:
                cmd = ""
            else:
                cmd += " && "
            cmd += f"printf '\\n$> {python_prefix + args}\\n\\n' && " + python_prefix + args
os.system(cmd)
