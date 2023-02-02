### PRECONDITIONS AND POSTCONDITIONS ###                                  # noqa: E266

# A precondition is something that must be true at the start of a
# function in order for it to work correctly.

# A postcondition is something that the function guarantees is true when it finishes.

## IMPORTS ##                                                             # noqa: E266

import numpy as np
import unittest as test
import uuid


def validate_key_in_dict_keys(input_key, dictionary, dict_name: str):
    assert input_key in dictionary, "Error; input key:{} is not in keys for dictionary:{}".format(input_key, dict_name)


def validate_value_in_list(input_value, list, list_name):
    assert input_value in list, "Error; input value:{} is not in list:{}".format(input_value, list_name) 


def validate_not_none(input, input_name):
    assert input is not None, "Error; {} is None.".format(input_name)


def validate_string_not_empty(input_string: str, string_name: str):
    assert bool(input_string) and bool(input_string.strip()), "Error: \
        {} is empty".format(string_name)


### DATA TYPES ###                                                         # noqa: E266

def is_int(n):
    return isinstance(n, (int, np.integer))


def validate_int(n):
    test.assertTrue(is_int(n), "Error: incorrect data type.\
        Integer is required.")


def is_float(n):
    return isinstance(n, (float, np.float))


def validate_float(n):
    test.assertTrue(is_float(n), "Error: incorrect data type.\
        Float is required.")


def is_uuid(n):
    return isinstance(n, uuid.UUID)


def validate_uuid(n):
    test.assertTrue(is_uuid(n), "Error, incorrect data type.\
        uuid is required.")


def is_bool(n):
    return isinstance(n, (bool, np.bool_))


def validate_bool(n):
    test.assertTrue(is_bool(n), "Error, incorrect data type.\
        Bool is required.")


def is_array(n):
    return isinstance(n, np.ndarray)


def validate_array(n):
    test.assertTrue(is_array(n), "Error, incorrect data type.\
        Array is required.")


### NUMERIC THRESHOLDS ###                                                # noqa: E266

def validate_non_negative_input(input_value, variable_name):
    assert input_value >= 0, "Error: {} must be non negative".format(variable_name)


def validate_greater_than_zero(input_value, variable_name):
    assert input_value > 0, "Error: {} must be greater than zero".format(variable_name)


def validate_above_threshold(input_value, threshold_value, variable_name):
    assert input_value > threshold_value, "Error: {} must \
        be greater than {}.".format(variable_name, threshold_value)


def validate_greater_or_equal_threshold(input_value, threshold_value, variable_name):
    assert input_value >= threshold_value, "Error: {} must \
        be greater or equal to {}.".format(variable_name, threshold_value)


def validate_below_threshold(input_value, threshold_value, variable_name):
    assert input_value < threshold_value, "Error: {} must be less than {}.".format(variable_name,
                                                                                   threshold_value)


def validate_below_or_equal_threshold(input_value, threshold_value, variable_name):
    assert input_value <= threshold_value, "Error: {} must \
        be less than or equal to {}.".format(variable_name, threshold_value)


def validate_equals_threshold(input_value, threshold_value, variable_name):
    assert input_value == threshold_value, "Error: {} must equal {}.".format(variable_name,
                                                                             threshold_value)
