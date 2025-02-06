import os
from src import util

def repeat(string, n):
    return ", ".join([string for _ in range(n)])

def json(size, num_fields, dims, increment=0):
    string = '''
{
    "meta": {
        "format": "1"
    },
    "steps": {
        "cedar.processing.sources.GaussInput": {
            "name": "Gauss Input",
            "dimensionality": "''' + str(dims) + '''",
            "sizes": [''' + repeat(f'"{size}"', dims) + '''],
            "amplitude": "1",
            "centers": [''' + repeat(f'"24"', dims) + '''],
            "sigma": [''' + repeat(f'"3"', dims) + '''],
            "cyclic": "false",
            "comments": ""
        },'''

    for i in range(num_fields):
        string += '''
        "cedar.dynamics.NeuralField": {
            "name": "Neural Field''' + str(i) + '''",
            "activation as output": "false",
            "discrete metric (workaround)": "false",
            "update stepIcon according to output": "true",
            "threshold for updating the stepIcon": "0.80000000000000004",
            "dimensionality": "''' + str(dims) + '''",
            "sizes": [''' + repeat(f'"{size}"', dims) + '''],
            "time scale": "100",
            "resting level": "''' + str(-0.7 + i * increment) + '''",
            "input noise gain": "''' + str(0.1+ i * increment) + '''",
            "multiplicative noise (input)": "false",
            "multiplicative noise (activation)": "false",
            "sigmoid": {
                "type": "cedar.aux.math.AbsSigmoid",
                "threshold": "0",
                "beta": "100"
            },
            "global inhibition": "''' + str(-0.01 + i * increment) + '''",
            "lateral kernels": {
                "cedar.aux.kernel.Gauss": {
                    "dimensionality": "''' + str(dims) + '''",
                    "anchor": [''' + repeat(f'"0"', dims) + '''],
                    "amplitude": "1",
                    "sigmas": [''' + repeat(f'"3"', dims) + '''],
                    "normalize": "true",
                    "shifts": [''' + repeat(f'"0"', dims) + '''],
                    "limit": "5"
                }
            },
            "lateral kernel convolution": {
                "engine": {
                    "type": "cedar.aux.conv.OpenCV"
                },
                "borderType": "Zero",
                "mode": "Same",
                "alternate even kernel center": "false"
            },
            "noise correlation kernel": {
                "dimensionality": "''' + str(dims) + '''",
                "anchor": [''' + repeat(f'"0"', dims) + '''],
                "amplitude": "0",
                "sigmas": [''' + repeat(f'"3"', dims) + '''],
                "normalize": "true",
                "shifts": [''' + repeat(f'"0"', dims) + '''],
                "limit": "5"
            },
            "comments": ""
        }'''
        if not i == num_fields - 1:
            string += ','
    
    string += '''
    },
    "triggers": {
        "cedar.processing.LoopedTrigger": {
            "name": "default thread",
            "step size": "0.02 s",
            "fake Euler step size": "0.02 s",
            "minimum sleep time": "0.0002 s",
            "idle time": "1e-05 s",
            "simulated time": "0.001 s",
            "loop mode": "fake deltaT",
            "use default CPU step": "true",
            "start with all": "true",
            "previous custom step size": "0.02 s",
            "listeners": [
'''
    for i in range(num_fields):
        string += f'                "Neural Field{i}"'
        if i != num_fields - 1:
            string += ','
            string += '\n'
    string += '''
            ]
        }
    },
    "connections": [
    '''
    for i in range(num_fields):
        string += '''
        {
            "source": "Gauss Input.Gauss input",
            "target": "Neural Field''' + str(i) + '''.input"
        }'''
        if i != num_fields - 1:
            string += ','
    string += '''
    ],
    "name": "element",
    "connectors": "",
    "is looped": "false",
    "time factor": "1",
    "loop mode": "fake deltaT",
    "simulation euler step": "0.02 s",
    "default CPU step": "0.02 s",
    "min computation time": "0.02 s",
    "ui": [
        {
            "type": "connections",
            "connections": ""
        },
        {
            "type": "step",
            "step": "Gauss Input",
            "display style": "ICON_ONLY",
            "width": "40",
            "height": "40",
            "positionX": "-854",
            "positionY": "-100"
        },'''
    for i in range(num_fields):
        y = -100 + i * 40
        string += '''
        {
            "type": "step",
            "step": "Neural Field''' + str(i) + '''",
            "display style": "ICON_AND_TEXT",
            "width": "124",
            "height": "40",
            "positionX": "-699",
            "positionY": "''' + str(y) + '''"
        }'''
        if i != num_fields - 1:
            string += ','
    string += ''',
        {
            "type": "trigger",
            "trigger": "default thread",
            "width": "30",
            "height": "30",
            "positionX": "0",
            "positionY": "0"
        }
    ],
    "ui view": {
        "ScrollBarX": "0",
        "ScrollBarY": "-215",
        "SliderPosX": "0",
        "SliderPosY": "-215",
        "Zoom": "1"
    },
    "ui generic": {
        "group": "element",
        "open plots": "",
        "plot groups": "",
        "architecture widgets": "",
        "robots": "",
        "width": "250",
        "height": "250",
        "smart mode": "false",
        "collapsed": "false",
        "lock geometry": "false",
        "uncollapsed width": "250",
        "uncollapsed height": "250",
        "positionX": "0",
        "positionY": "0"
    }
}
    '''
    return string

for dim in range(2, 5):
    suffix = "" if dim == 2 else f"_{dim}D"
    size = 50
    num_fields = 3
    with open(os.path.join(util.root(), f"architectures/perf_0{suffix}.json"), "w") as f:
        f.write(json(size, num_fields, dim))

    size = 50
    num_fields = 300
    with open(os.path.join(util.root(), f"architectures/perf_1{suffix}.json"), "w") as f:
        f.write(json(size, num_fields, dim))

size = 500
num_fields = 30
dim = 2
with open(os.path.join(util.root(), "architectures/perf_2.json"), "w") as f:
    f.write(json(size, num_fields, dim))

size = 50
num_fields = 300
dim = 2
with open(os.path.join(util.root(), "architectures/perf_3.json"), "w") as f:
    f.write(json(size, num_fields, dim))

size = 500
num_fields = 30
dim = 2
with open(os.path.join(util.root(), "architectures/perf_4.json"), "w") as f:
    f.write(json(size, num_fields, dim, increment=0.0001))