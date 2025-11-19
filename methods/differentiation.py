def forward_diff(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=1e-5):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
