"""
code_chunking.py

Implements the chunking approach from your snippet:
 - generate_and_chunk_code, display_code_chunk, get_next_chunk, get_previous_chunk
 - possibly also the AST merges
"""

import ast

class CodeChunker:
    def __init__(self, chunk_size=1000, max_lines=15000):
        self.chunk_size = chunk_size
        self.max_lines = max_lines
        self.code_buffer = []

    def generate_and_chunk_code(self, full_code: str):
        lines = full_code.split('\n')
        if len(lines) > self.max_lines:
            lines = lines[:self.max_lines]
        self.code_buffer = [lines[i:i+self.chunk_size] for i in range(0, len(lines), self.chunk_size)]

    def display_code_chunk(self, chunk_index):
        if 0 <= chunk_index < len(self.code_buffer):
            chunk = self.code_buffer[chunk_index]
            return '\n'.join(chunk), chunk_index, len(self.code_buffer)
        else:
            return "No more chunks.", -1, len(self.code_buffer)

    def get_next_chunk(self, current_chunk):
        return self.display_code_chunk(current_chunk + 1)

    def get_previous_chunk(self, current_chunk):
        return self.display_code_chunk(current_chunk - 1)

    def merge_asts(self, current_ast, new_ast):
        # example
        for node in new_ast.body:
            if isinstance(node, ast.ClassDef) and node.name == 'JAIRVISMKIV':
                # find the same class in current_ast
                ...
        return current_ast
