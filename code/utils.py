## From official MATH dataset repo
import torch
import numpy as np
import pprint

from math_equivalence import is_equiv

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def pad(tensors, padding_value = 0, padding_side = "right"):
    """
    Pads a list of tensors to the same shape along the first dimension.
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

def check_correct(completions, solutions):
        """
        Check correctness of each completion by comparing it to the solution.
        Returns a list 
        """        
        correct = []
        for completion, solution in zip(completions, solutions):

            model_answer = remove_boxed(last_boxed_only_string(completion))
            correct_answer = remove_boxed(last_boxed_only_string(solution))

            if is_equiv(model_answer, correct_answer):
                correct.append(1)
            else:
                correct.append(0)

        return correct