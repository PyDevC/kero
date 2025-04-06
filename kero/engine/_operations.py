import torch

# Aggregation Operations

def tensor_sum(tensor, dim=None):
    return torch.sum(tensor, dim=dim)

def tensor_mean(tensor, dim=None):
    return torch.mean(tensor, dim=dim)

def tensor_median(tensor, dim=None):
    return torch.median(tensor, dim=dim)

def tensor_prod(tensor, dim=None):
    return torch.prod(tensor, dim=dim)

def tensor_std(tensor, dim=None):
    return torch.std(tensor, dim=dim)

def tensor_var(tensor, dim=None):
    return torch.var(tensor, dim=dim)

def tensor_min(tensor, dim=None):
    return torch.min(tensor, dim=dim)

def tensor_max(tensor, dim=None):
    return torch.max(tensor, dim=dim)

def tensor_argmax(tensor, dim=None):
    return torch.argmax(tensor, dim=dim)

def tensor_argmin(tensor, dim=None):
    return torch.argmin(tensor, dim=dim)


# Mathemetical Operations

def tensor_add(tensor1, tensor2):
    return torch.add(tensor1, tensor2)

def tensor_sub(tensor1, tensor2):
    return torch.sub(tensor1, tensor2)

def tensor_mul(tensor1, tensor2):
    return torch.mul(tensor1, tensor2)

def tensor_div(tensor1, tensor2):
    return torch.div(tensor1, tensor2)

def tensor_pow(tensor, exp):
    return torch.pow(tensor, exp)

def tensor_exp(tensor):
    return torch.exp(tensor)

def tensor_log(tensor):
    return torch.log(tensor)

def tensor_sqrt(tensor):
    return torch.sqrt(tensor)

def tensor_abs(tensor):
    return torch.abs(tensor)

def tensor_ceil(tensor):
    return torch.ceil(tensor)

def tensor_floor(tensor):
    return torch.floor(tensor)

def tensor_round(tensor):
    return torch.round(tensor)

def tensor_clamp(tensor):
    return torch.clamp(tensor)

# Logical Operations

def tensor_equal(tensor1, tensor2):
    return torch.eq(tensor1, tensor2)

def tensor_not_equal(tensor1, tensor2):
    return torch.ne(tensor1, tensor2)

def tensor_greater_than(tensor1, tensor2):
    return torch.gt(tensor1, tensor2)

def tensor_greater_than_equalto(tensor1, tensor2):
    return torch.ge(tensor1, tensor2)

def tensor_less_than(tensor1, tensor2):
    return torch.lt(tensor1, tensor2)

def tensor_less_than_equalto(tensor1, tensor2):
    return torch.le(tensor1, tensor2)

def tensor_and(tensor1, tensor2):
    return torch.logical_and(tensor1, tensor2)

def tensor_or(tensor1, tensor2):
    return torch.logical_or(tensor1, tensor2)

def tensor_not(tensor):
    return torch.logical_not(tensor)

def tensor_where(tensor1, tensor2, condition):
    return torch.where(condition, tensor1, tensor2)
