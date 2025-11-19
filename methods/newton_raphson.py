def newton_raphson(f, x0, tol=1e-6, max_iter=50):
    def df(x):
        h = 1e-6
        return (f(x + h) - f(x - h)) / (2 * h)

    x = x0
    iterations = []

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            return {"error": "Derivative too small"}

        x_next = x - fx / dfx
        iterations.append({"iter": i + 1, "x": x_next, "f(x)": f(x_next)})

        if abs(x_next - x) < tol or abs(fx) < tol:
            return {"method": "Newton-Raphson", "root": x_next, "iterations": iterations}

        x = x_next

    return {"method": "Newton-Raphson", "root": x, "iterations": iterations, "converged": False}
