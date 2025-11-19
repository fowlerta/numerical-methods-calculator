def bisection(f, a, b, tol=1e-6, max_iter=50):
    if f(a) * f(b) >= 0:
        return {"error": "Invalid interval: f(a) and f(b) must have opposite signs."}

    iterations = []
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        iterations.append({"iter": i + 1, "a": a, "b": b, "c": c, "f(c)": fc})

        if abs(fc) < tol or (b - a) / 2 < tol:
            return {"method": "Bisection", "root": c, "iterations": iterations}

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return {"method": "Bisection", "root": c, "iterations": iterations, "converged": False}
