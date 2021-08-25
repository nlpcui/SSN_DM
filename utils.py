import sys, json, os, torch, math
import numpy as np

def binary_search(array, begin, end, target):
    if begin > end:
        return -1
    pivot = (begin + end ) // 2
    if array[pivot] == target:
        return pivot
    elif array[pivot] < target:
        return binary_search(array, pivot+1, end)
    else:
        return binary_search(array, begin, pivot-1)


def check_nan(tensor):
    grad_nan_cnt = 0
    value_nan_cnt = torch.sum(torch.isnan(tensor))
    if tensor.grad is not None:
        grad_nan_cnt = torch.sum(torch.isnan(tensor.grad))
    
    return value_nan_cnt, grad_nan_cnt, tensor.numel()



def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))