"""Tools the reasoning step can call.

`calculator` evaluates arithmetic safely (AST walk, no eval) so the LLM never does maths in its head.
`submit_answer` is how the model hands in its final answer together with the pages it relied on.
"""

from __future__ import annotations

import ast
import math
import operator
import re

from langchain_core.tools import tool

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


def safe_eval(expression: str) -> float:
    # Tolerate how numbers appear in reports: "$1,577", "(370)" stays valid as a parenthesised number.
    expr = expression.replace("$", "").replace("%", "").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            right = ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")

    return float(ev(tree))


def format_number(x: float) -> str:
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.6g}" if abs(x) < 1e-3 else f"{round(x, 4)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Use this for EVERY calculation (sums, differences, ratios, margins, growth rates).
    Write plain numbers without units, e.g. "(1577 - 1373) / 1373 * 100".
    Supports + - * / ** %, parentheses, abs(), round(), min(), max(), sqrt().
    """
    try:
        return format_number(safe_eval(expression))
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception as e:
        return f"error: {e}"


@tool
def submit_answer(answer: str, pages: list[int]) -> str:
    """Submit the final answer.

    answer: a direct, concise answer to the question, including the number(s) with units
            (e.g. "$1,577 million") and one sentence on how it was derived.
    pages:  the report page numbers (as shown in the [doc | page N] headers) the answer relies on.
    """
    return "submitted"
