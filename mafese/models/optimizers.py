#!/usr/bin/env python
# Created by "Thieu" at 06:50, 04/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import *
from mealpy.bio_based import *
from mealpy.math_based import *
from mealpy.swarm_based import *
from mealpy.physics_based import *
from mealpy.human_based import *
from mealpy.system_based import *
from mealpy.music_based import *
import sys, inspect

EXCLUDE_MODULES = ["__builtins__", "current_module", "inspect", "sys"]


def get_all_classes():
    cls = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.ismodule(obj) and (name not in EXCLUDE_MODULES):
            for cls_name, cls_obj in inspect.getmembers(obj):
                if inspect.isclass(cls_obj):
                    cls[cls_name] = cls_obj
    del cls['Optimizer']
    return cls


def get_optimizer(name):
    try:
        cls = get_all_classes()[name]
        return cls
    except KeyError:
        print(f"Mealpy doesn't support optimizer named: {name}.\n"
              f"Please see the optimizer's classname from here: https://mealpy.readthedocs.io/en/latest/pages/support.html#classification-table")
