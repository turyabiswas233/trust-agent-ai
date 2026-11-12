import ast
import math
import operator

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_ABS_VALUE = 1e100


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude must not exceed 100.")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        value = _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    else:
        raise ValueError("Only numeric constants and basic arithmetic are supported.")

    if type(value) not in (int, float):
        raise ValueError("Calculation result must be a real number.")
    if not math.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
        raise ValueError("Calculation result is outside the supported numeric range.")
    return value


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic without executing Python code."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except (SyntaxError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"Invalid arithmetic expression: {error}") from error
    return str(result)
