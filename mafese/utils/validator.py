#!/usr/bin/env python
# Created by "Thieu" at 21:39, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


import operator
import numpy as np
from numbers import Number


SEQUENCE = (list, tuple, np.ndarray)
DIGIT = (int, np.integer)
REAL = (float, np.floating)


def is_in_bound(value, bound):
    """
    Checks if a value falls within a specified numerical bound.

    Args:
        value (float): The value to check.
        bound (tuple): A tuple representing the lower and upper bound (inclusive for lists).

    Returns:
        bool: True if the value is within the bound, False otherwise.

    Raises:
        ValueError: If the bound is not a tuple or list.
    """
    ops = None
    if type(bound) is tuple:
        ops = operator.lt
    elif type(bound) is list:
        ops = operator.le
    if bound[0] == float("-inf") and bound[1] == float("inf"):
        return True
    elif bound[0] == float("-inf") and ops(value, bound[1]):
        return True
    elif ops(bound[0], value) and bound[1] == float("inf"):
        return True
    elif ops(bound[0], value) and ops(value, bound[1]):
        return True
    return False


def is_str_in_list(value: str, my_list: list):
    """
    Checks if a string value exists within a provided list.

    Args:
        value (str): The string value to check.
        my_list (list, optional): The list of possible values.

    Returns:
        bool: True if the value is in the list, False otherwise.
    """
    if type(value) == str and my_list is not None:
        return True if value in my_list else False
    return False


def check_int(name: str, value: None, bound=None):
    """
    Checks if a value is an integer and optionally verifies it falls within a specified bound.

    Args:
        name (str): The name of the variable being checked.
        value (int or float): The value to check.
        bound (tuple, optional): A tuple representing the lower and upper bound (inclusive).

    Returns:
        int: The validated integer value.

    Raises:
        ValueError: If the value is not an integer or falls outside the bound (if provided).
    """
    if isinstance(value, Number):
        if bound is None:
            return int(value)
        elif is_in_bound(value, bound):
            return int(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is an integer {bound}.")


def check_float(name: str, value: None, bound=None):
    """
    Checks if a value is a float and optionally verifies it falls within a specified bound.

    Args:
        name (str): The name of the variable being checked.
        value (int or float): The value to check.
        bound (tuple, optional): A tuple representing the lower and upper bound (inclusive).

    Returns:
        float: The validated float value.

    Raises:
        ValueError: If the value is not a float or falls outside the bound (if provided).
    """
    if isinstance(value, Number):
        if bound is None:
            return float(value)
        elif is_in_bound(value, bound):
            return float(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is a float {bound}.")


def check_str(name: str, value: str, bound=None):
    """
    Checks if a value is a string and optionally verifies it exists within a provided list.

    Args:
        name (str): The name of the variable being checked.
        value (str): The value to check.
        bound (list, optional): A list of allowed string values.

    Returns:
        str: The validated string value.

    Raises:
        ValueError: If the value is not a string or not found in the bound list (if provided).
    """
    if type(value) is str:
        if bound is None or is_str_in_list(value, bound):
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a string {bound}.")


def check_bool(name: str, value: bool, bound=(True, False)):
    """
    Checks if a value is a boolean and optionally verifies it matches a specified bound.

    Args:
        name (str): The name of the variable being checked.
        value (bool): The value to check.
        bound (tuple, optional): A tuple of allowed boolean values.

    Returns:
        bool: The validated boolean value.

    Raises:
        ValueError: If the value is not a boolean or not in the bound (if provided).
    """
    if type(value) is bool:
        if value in bound:
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a boolean {bound}.")


def check_tuple_int(name: str, values: None, bounds=None):
    """
    Checks if a tuple contains only integers and optionally verifies they fall within specified bounds.

    Args:
        name (str): The name of the variable being checked.
        values (tuple): The tuple of values to check.
        bounds (list of tuples, optional): A list of tuples representing lower and upper bounds for each value.

    Returns:
        tuple: The validated tuple of integers.

    Raises:
        ValueError: If the values are not all integers or do not fall within the specified bounds.
    """
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, DIGIT) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are integer {bounds}.")


def check_tuple_float(name: str, values: tuple, bounds=None):
    """
    Checks if a tuple contains only floats or integers and optionally verifies they fall within specified bounds.

    Args:
        name (str): The name of the variable being checked.
        values (tuple): The tuple of values to check.
        bounds (list of tuples, optional): A list of tuples representing lower and upper bounds for each value.

    Returns:
        tuple: The validated tuple of floats.

    Raises:
        ValueError: If the values are not all floats or integers or do not fall within the specified bounds.
    """
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, Number) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are float {bounds}.")
