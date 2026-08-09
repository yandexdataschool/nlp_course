import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: float, b: float) -> float:
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

def _to_vector(x):
    arr = np.array(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D list representing a vector.")
    return arr


def _to_matrix(x):
    arr = np.array(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Input must be a 2D list representing a matrix.")
    return arr


@mcp.tool()
def vector_add(a: list, b: list) -> list:
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return (va + vb).tolist()


@mcp.tool()
def vector_subtract(a: list, b: list) -> list:
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return (va - vb).tolist()


@mcp.tool()
def vector_dot(a: list, b: list) -> float:
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return float(np.dot(va, vb))


@mcp.tool()
def vector_elementwise_multiply(a: list, b: list) -> list:
    va, vb = _to_vector(a), _to_vector(b)
    if va.shape != vb.shape:
        raise ValueError("Vectors must have the same size.")
    return (va * vb).tolist()


@mcp.tool()
def matrix_add(a: list, b: list) -> list:
    ma, mb = _to_matrix(a), _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError("Matrices must have the same shape.")
    return (ma + mb).tolist()


@mcp.tool()
def matrix_subtract(a: list, b: list) -> list:
    ma, mb = _to_matrix(a), _to_matrix(b)
    if ma.shape != mb.shape:
        raise ValueError("Matrices must have the same shape.")
    return (ma - mb).tolist()


@mcp.tool()
def matrix_multiply(a: list, b: list) -> list:
    ma, mb = _to_matrix(a), _to_matrix(b)
    try:
        result = ma @ mb
    except ValueError:
        raise ValueError(
            f"Incompatible matrix shapes for multiplication: {ma.shape} x {mb.shape}"
        )
    return result.tolist()


@mcp.tool()
def matrix_transpose(a: list) -> list:
    ma = _to_matrix(a)
    return ma.T.tolist()


if __name__ == "__main__":
    mcp.run(transport="stdio")
