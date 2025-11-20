import os
from flask import Flask, render_template, request, jsonify
from sympy import sympify, Symbol, lambdify, sin, cos, tan, exp, log, sqrt, pi, E
import numpy as np

app = Flask(__name__)
x_sym = Symbol('x')

# Supported functions
allowed_funcs = {
    'sin': sin, 'cos': cos, 'tan': tan,
    'exp': exp, 'log': log, 'sqrt': sqrt,
    'pi': pi, 'e': E, 'E': E
}

def safe_func(expr_str):
    try:
        expr = sympify(expr_str, locals=allowed_funcs)
        f = lambdify(x_sym, expr, modules=['numpy', 'math'])
        return f, expr
    except Exception as e:
        raise ValueError(f"Invalid function: {str(e)}. Use x as variable and functions like sin(x), pi, e, etc.")

# Import methods
from methods.bisection import bisection
from methods.newton_raphson import newton_raphson
from methods.lagrange import lagrange_interpolation
from methods.differentiation import forward_diff, backward_diff, central_diff
from methods.integration import trapezoidal, simpson_one_third, simpson_three_eighth

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    method = data.get('method')
    func_str = data.get('function', '').strip()

    if not func_str:
        return jsonify({"error": "Function is required"}), 400

    try:
        f, expr = safe_func(func_str)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    tol = float(data.get('tolerance', 1e-6))
    max_iter = int(data.get('max_iter', 100))
    result = {}
    plot_data = {"x": [], "y": [], "extra": []}

    try:
        # === ROOT FINDING ===
        if method == "newton":
            x0 = float(data['x0'])
            res = newton_raphson(f, x0, tol, max_iter)
            result = {**res, "method": "Newton-Raphson"}
            root = res.get("root")
            if root is not None:
                plot_data["extra"].append({"type": "point", "x": root, "y": f(root), "label": "Root"})

        elif method == "bisection":
            a, b = float(data['a']), float(data['b'])
            res = bisection(f, a, b, tol, max_iter)
            result = {**res, "method": "Bisection"}
            root = res.get("root")
            if root is not None:
                plot_data["extra"].append({"type": "point", "x": root, "y": f(root), "label": "Root"})
                plot_data["extra"].extend([
                    {"type": "vline", "x": a},
                    {"type": "vline", "x": b}
                ])

        # === LAGRANGE ===
        elif method == "lagrange":
            xs = [float(x) for x in data['x_points'].split(',')]
            ys = [float(y) for y in data['y_points'].split(',')]
            x_eval = float(data['x_eval'])
            value = lagrange_interpolation(xs, ys, x_eval)
            result = {"method": "Lagrange Interpolation", "value": value}
            plot_data["extra"] = [{"type": "point", "x": x, "y": y, "label": "Data"} for x, y in zip(xs, ys)]
            plot_data["extra"].append({"type": "point", "x": x_eval, "y": value, "label": "P(x)"})

        # === DIFFERENTIATION ===
        elif method == "diff":
            x = float(data['x'])
            h = 1e-5
            diff_method = data.get("diff_method", "central")
            if diff_method == "forward":
                value = forward_diff(f, x, h)
            elif diff_method == "backward":
                value = backward_diff(f, x, h)
            else:
                value = central_diff(f, x, h)
            result = {"method": f"Numerical Differentiation ({diff_method})", "value": value}
            plot_data["extra"].append({"type": "tangent", "x": x, "y": f(x), "slope": value})

        # === INTEGRATION ===
        elif method in ["trap", "simpson13", "simpson38"]:
            a, b = float(data['a']), float(data['b'])
            n = int(data['n'])

            if method == "trap":
                value = trapezoidal(f, a, b, n)
                result = {"method": "Trapezoidal Rule", "result": value}
            elif method == "simpson13":
                if n % 2 != 0: n += 1
                value = simpson_one_third(f, a, b, n)
                result = {"method": "Simpson 1/3 Rule", "result": value}
            else:  # simpson38
                if n % 3 != 0: n = ((n // 3) + 1) * 3
                value = simpson_three_eighth(f, a, b, n)
                result = {"method": "Simpson 3/8 Rule", "result": value}

        # === PLOT DATA ===
        xmin, xmax = -10, 10
        if method in ["trap", "simpson13", "simpson38"]:
            xmin, xmax = a, b
        elif method == "lagrange":
            all_x = xs + [x_eval]
            xmin, xmax = min(all_x) - 1, max(all_x) + 1

        x_vals = np.linspace(xmin, xmax, 500)
        y_vals = [f(x) for x in x_vals]

        plot_data["x"] = x_vals.tolist()
        plot_data["y"] = y_vals
        result["plot"] = plot_data

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# PERFECT FOR BOTH LOCAL + RENDER
if __name__ == '__main__':
    # Local development
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # Render / production
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
