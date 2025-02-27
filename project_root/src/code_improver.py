"""
code_improver.py

Contains the CodeImprover class for analyzing/improving code
(AST transformations, etc.).
"""

import ast

class CodeImprover(ast.NodeTransformer):
    """Used to demonstrate self-analysis & code improvement."""
    def __init__(self, improvements):
        self.improvements = improvements

    def visit(self, node):
        node = self.generic_visit(node)
        if isinstance(node, ast.FunctionDef):
            node = self.improve_function(node)
        elif isinstance(node, ast.ClassDef):
            node = self.improve_class(node)
        elif isinstance(node, ast.Assign):
            node = self.improve_assignment(node)
        return node

    def improve_function(self, node):
        # Add docstring if missing
        if not ast.get_docstring(node):
            node.body.insert(0, ast.Expr(ast.Str("Auto-generated docstring")))

        # Add type hints if missing
        if not node.returns:
            node.returns = ast.Name(id='Any', ctx=ast.Load())
        for arg in node.args.args:
            if not arg.annotation:
                arg.annotation = ast.Name(id='Any', ctx=ast.Load())
        return node

    def improve_class(self, node):
        # Add class docstring if missing
        if not ast.get_docstring(node):
            node.body.insert(0, ast.Expr(ast.Str("Auto-generated class docstring")))
        return node

    def improve_assignment(self, node):
        # Add type comments for assignments
        if not hasattr(node, 'type_comment'):
            node.type_comment = '# type: Any'
        return node
