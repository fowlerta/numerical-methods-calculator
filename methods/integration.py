def trapezoidal(f, a, b, n=100):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        total += f(a + i * h)

    return total * h


def simpson_one_third(f, a, b, n=100):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson 1/3")

    h = (b - a) / n
    total = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        total += (4 if i % 2 != 0 else 2) * f(x)

    return total * h / 3


def simpson_three_eighth(f, a, b, n=100):
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3")

    h = (b - a) / n
    total = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h
        weight = 3 if i % 3 != 0 else 2
        total += weight * f(x)

    return total * 3 * h / 8
