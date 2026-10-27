# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Lexer and recursive descent parser for a practical subset of VEX.

The parser produces a small AST made of ``Node`` objects which the SIMD interpreter in
``interp.py`` executes over all elements of a geometry at once.
"""

import re

TYPE_NAMES = {
    'int', 'float', 'vector', 'vector2', 'vector4', 'matrix', 'matrix2', 'matrix3', 'string', 'void', 'dict', 'bsdf',
}

# @-binding prefixes.
ATTR_PREFIX = {
    'f': 'float', 'i': 'int', 'v': 'vector', 'p': 'vector4', 'u': 'vector2', 's': 'string',
    '2': 'matrix2', '3': 'matrix3', '4': 'matrix', 'd': 'dict',
}


class VexSyntaxError(Exception):
    def __init__(self, msg, line=0, col=0):
        super().__init__("Line %d: %s" % (line, msg) if line else msg)
        self.line = line
        self.col = col


class Node:
    __slots__ = ('kind', 'a', 'line')

    def __init__(self, kind, line=0, **a):
        self.kind = kind
        self.a = a
        self.line = line

    def __getattr__(self, name):
        try:
            return self.a[name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        return "%s(%s)" % (self.kind, ", ".join("%s=%r" % kv for kv in self.a.items()))


# ----------------------------------------------------------------------------
# Lexer

class Tok:
    __slots__ = ('type', 'value', 'line', 'col')

    def __init__(self, type, value, line, col):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return "Tok(%s, %r)" % (self.type, self.value)


_OPS = sorted([
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '==', '!=', '<=', '>=', '&&', '||', '++', '--',
    '<<', '>>', '+', '-', '*', '/', '%', '=', '<', '>', '!', '?', ':', ';', ',', '.', '(', ')', '[', ']',
    '{', '}', '&', '|', '^', '~',
], key=len, reverse=True)

_num_re = re.compile(r'(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[fF]?')
_ident_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_attr_re = re.compile(r'(?:([fivpus234d])(\[\])?)?@([A-Za-z_][A-Za-z0-9_]*)')


def tokenize(src, macros=None):
    toks = []
    macros = {} if macros is None else macros
    i = 0
    n = len(src)
    line = 1
    line_start = 0
    while i < n:
        c = src[i]
        if c == '\n':
            line += 1
            i += 1
            line_start = i
            continue
        if c in ' \t\r':
            i += 1
            continue
        col = i - line_start + 1
        if src.startswith('//', i):
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if src.startswith('/*', i):
            j = src.find('*/', i + 2)
            if j < 0:
                raise VexSyntaxError("Unterminated comment", line, col)
            line += src.count('\n', i, j)
            i = j + 2
            continue
        if c == '#' and src[line_start:i].strip() == '':
            # Preprocessor: object-like #define macros are expanded, other directives ignored.
            j = src.find('\n', i)
            directive = src[i:n if j < 0 else j]
            m = re.match(r'#\s*define\s+([A-Za-z_]\w*)(?!\()\s*(.*)$', directive)
            if m:
                macros[m.group(1)] = tokenize(m.group(2), macros)[:-1] if m.group(2).strip() else []
            i = n if j < 0 else j
            continue
        # Attribute bindings: @P, v@foo, i[]@bar, 4@xform
        if c == '@' or (c in 'fivpus234d' and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == '_'))):
            m = _attr_re.match(src, i)
            if m:
                prefix, arr, name = m.groups()
                toks.append(Tok('attr', (prefix, bool(arr), name), line, col))
                i = m.end()
                continue
        if c.isdigit() or (c == '.' and i + 1 < n and src[i + 1].isdigit()):
            m = _num_re.match(src, i)
            text = m.group(0)
            if '.' in text or 'e' in text.lower() or text.lower().endswith('f'):
                toks.append(Tok('float', float(text.rstrip('fF')), line, col))
            else:
                toks.append(Tok('int', int(text), line, col))
            i = m.end()
            continue
        if c.isalpha() or c == '_':
            m = _ident_re.match(src, i)
            word = m.group(0)
            if word in macros:
                for t in macros[word]:
                    toks.append(Tok(t.type, t.value, line, col))
            else:
                toks.append(Tok('ident', word, line, col))
            i = m.end()
            continue
        if c in '"\'':
            j = i + 1
            out = []
            while j < n and src[j] != c:
                if src[j] == '\\' and j + 1 < n:
                    esc = src[j + 1]
                    out.append({'n': '\n', 't': '\t', '\\': '\\', '"': '"', "'": "'"}.get(esc, '\\' + esc))
                    j += 2
                    continue
                if src[j] == '\n':
                    raise VexSyntaxError("Unterminated string", line, col)
                out.append(src[j])
                j += 1
            toks.append(Tok('string', ''.join(out), line, col))
            i = j + 1
            continue
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Tok('op', op, line, col))
                i += len(op)
                break
        else:
            raise VexSyntaxError("Unexpected character %r" % c, line, col)
    toks.append(Tok('eof', None, line, 0))
    return toks


# ----------------------------------------------------------------------------
# Parser

_ASSIGN_OPS = {'=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^='}

_BINARY_PREC = [
    ('||',),
    ('&&',),
    ('|',),
    ('^',),
    ('&',),
    ('==', '!='),
    ('<', '>', '<=', '>='),
    ('<<', '>>'),
    ('+', '-'),
    ('*', '/', '%'),
]


class Parser:
    def __init__(self, src):
        self.toks = tokenize(src)
        self.pos = 0
        self.functions = {}
        self.bindings = {}  # name -> declared type (None if implicit)

    # -- token helpers

    @property
    def tok(self):
        return self.toks[self.pos]

    def peek(self, k=1):
        return self.toks[min(self.pos + k, len(self.toks) - 1)]

    def next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def is_op(self, op, tok=None):
        tok = tok or self.tok
        return tok.type == 'op' and tok.value == op

    def accept(self, op):
        if self.is_op(op):
            self.pos += 1
            return True
        return False

    def expect(self, op):
        if not self.is_op(op):
            t = self.tok
            raise VexSyntaxError("Expected '%s' but found %s" % (op, repr(t.value) if t.value is not None else 'end of code'),
                                 t.line, t.col)
        return self.next()

    def expect_ident(self):
        t = self.tok
        if t.type != 'ident':
            raise VexSyntaxError("Expected identifier, found %r" % (t.value,), t.line, t.col)
        self.pos += 1
        return t.value

    def is_type(self, tok=None):
        tok = tok or self.tok
        return tok.type == 'ident' and tok.value in TYPE_NAMES

    # -- program

    def parse_program(self):
        stmts = []
        while self.tok.type != 'eof':
            fn = self.try_function_def()
            if fn is not None:
                self.functions[fn.name] = fn
                continue
            stmts.append(self.parse_statement())
        return Node('program', 0, body=stmts, functions=self.functions, bindings=self.bindings)

    def try_function_def(self):
        start = self.pos
        if self.tok.type == 'ident' and self.tok.value == 'function':
            self.pos += 1
        if not self.is_type():
            self.pos = start
            return None
        rtype = self.next().value
        if self.is_op('[') and self.is_op(']', self.peek()):
            self.pos += 2
            rtype += '[]'
        if self.tok.type == 'ident' and self.is_op('(', self.peek()) and self.tok.value not in TYPE_NAMES:
            line = self.tok.line
            name = self.next().value
            self.expect('(')
            params = []
            while not self.is_op(')'):
                while self.tok.type == 'ident' and self.tok.value in ('const', 'export'):
                    self.next()
                if not self.is_type():
                    raise VexSyntaxError("Expected parameter type", self.tok.line, self.tok.col)
                ptype = self.next().value
                while True:
                    pname = self.expect_ident()
                    t = ptype
                    if self.accept('['):
                        self.expect(']')
                        t += '[]'
                    params.append((t, pname))
                    if self.accept(','):
                        if self.is_type():
                            break
                        continue
                    break
                if not self.accept(';'):
                    if not self.is_op(')') and not self.is_type():
                        self.expect(')')
            self.expect(')')
            body = self.parse_block()
            return Node('function', line, name=name, rtype=rtype, params=params, body=body)
        self.pos = start
        return None

    # -- statements

    def parse_block(self):
        self.expect('{')
        stmts = []
        while not self.is_op('}'):
            if self.tok.type == 'eof':
                raise VexSyntaxError("Missing '}'", self.tok.line, self.tok.col)
            stmts.append(self.parse_statement())
        self.expect('}')
        return Node('block', 0, body=stmts)

    def parse_statement(self):
        t = self.tok
        line = t.line
        if self.is_op('{'):
            return self.parse_block()
        if self.is_op(';'):
            self.next()
            return Node('block', line, body=[])
        if t.type == 'ident':
            kw = t.value
            if kw == 'if':
                self.next()
                self.expect('(')
                cond = self.parse_expr()
                self.expect(')')
                then = self.parse_statement()
                other = None
                if self.tok.type == 'ident' and self.tok.value == 'else':
                    self.next()
                    other = self.parse_statement()
                return Node('if', line, cond=cond, then=then, other=other)
            if kw == 'while':
                self.next()
                self.expect('(')
                cond = self.parse_expr()
                self.expect(')')
                body = self.parse_statement()
                return Node('for', line, init=None, cond=cond, step=None, body=body, do=False)
            if kw == 'do':
                self.next()
                body = self.parse_statement()
                if not (self.tok.type == 'ident' and self.tok.value == 'while'):
                    raise VexSyntaxError("Expected 'while' after do-block", self.tok.line, self.tok.col)
                self.next()
                self.expect('(')
                cond = self.parse_expr()
                self.expect(')')
                self.accept(';')
                return Node('for', line, init=None, cond=cond, step=None, body=body, do=True)
            if kw == 'for':
                self.next()
                self.expect('(')
                init = None
                if not self.is_op(';'):
                    init = self.parse_decl_or_expr_stmt(need_semicolon=False)
                self.expect(';')
                cond = None if self.is_op(';') else self.parse_expr()
                self.expect(';')
                step = None if self.is_op(')') else self.parse_expr_list()
                self.expect(')')
                body = self.parse_statement()
                return Node('for', line, init=init, cond=cond, step=step, body=body, do=False)
            if kw == 'foreach':
                self.next()
                self.expect('(')
                vtype = None
                if self.is_type():
                    vtype = self.next().value
                names = [self.expect_ident()]
                if self.accept(',') or (self.is_op(';') and (self.is_type(self.peek()) or self.is_op(';', self.peek(2)))):
                    self.accept(';')
                    if self.is_type():
                        vtype = self.next().value
                    names.append(self.expect_ident())
                self.expect(';')
                arr = self.parse_expr()
                self.expect(')')
                body = self.parse_statement()
                if len(names) == 2:
                    idx_name, var_name = names
                else:
                    idx_name, var_name = None, names[0]
                return Node('foreach', line, vtype=vtype, var=var_name, index=idx_name, array=arr, body=body)
            if kw == 'break':
                self.next()
                self.accept(';')
                return Node('break', line)
            if kw == 'continue':
                self.next()
                self.accept(';')
                return Node('continue', line)
            if kw == 'return':
                self.next()
                value = None if self.is_op(';') else self.parse_expr()
                self.accept(';')
                return Node('return', line, value=value)
        stmt = self.parse_decl_or_expr_stmt()
        return stmt

    def parse_expr_list(self):
        exprs = [self.parse_expr()]
        while self.accept(','):
            exprs.append(self.parse_expr())
        return exprs[0] if len(exprs) == 1 else Node('seq', exprs[0].line, items=exprs)

    def parse_decl_or_expr_stmt(self, need_semicolon=True):
        line = self.tok.line
        is_decl = False
        if self.tok.type == 'ident' and self.tok.value == 'const':
            self.next()
            is_decl = True
        if self.is_type() and not self.is_op('(', self.peek()):
            is_decl = True
        if is_decl:
            base = self.next().value
            arr_all = False
            if self.is_op('[') and self.is_op(']', self.peek()):
                self.pos += 2
                arr_all = True
            decls = []
            while True:
                name = self.expect_ident()
                t = base + ('[]' if arr_all else '')
                if self.accept('['):
                    # int a[]; or int a[3];
                    if not self.is_op(']'):
                        self.parse_expr()
                    self.expect(']')
                    t = base + '[]'
                init = None
                if self.accept('='):
                    init = self.parse_assign()
                decls.append((t, name, init))
                if not self.accept(','):
                    break
            if need_semicolon:
                self.expect(';')
            return Node('decl', line, decls=decls)
        expr = self.parse_expr_list()
        if need_semicolon:
            self.expect(';')
        return Node('expr', line, expr=expr)

    # -- expressions

    def parse_expr(self):
        return self.parse_assign()

    def parse_assign(self):
        left = self.parse_ternary()
        if self.tok.type == 'op' and self.tok.value in _ASSIGN_OPS:
            op = self.next().value
            right = self.parse_assign()
            if left.kind not in ('var', 'attr', 'member', 'index'):
                raise VexSyntaxError("Can't assign to this expression", left.line)
            return Node('assign', left.line, op=op, target=left, value=right)
        return left

    def parse_ternary(self):
        cond = self.parse_binary(0)
        if self.accept('?'):
            a = self.parse_assign()
            self.expect(':')
            b = self.parse_assign()
            return Node('ternary', cond.line, cond=cond, yes=a, no=b)
        return cond

    def parse_binary(self, level):
        if level >= len(_BINARY_PREC):
            return self.parse_unary()
        left = self.parse_binary(level + 1)
        ops = _BINARY_PREC[level]
        while self.tok.type == 'op' and self.tok.value in ops:
            op = self.next().value
            right = self.parse_binary(level + 1)
            left = Node('binop', left.line, op=op, l=left, r=right)
        return left

    def parse_unary(self):
        t = self.tok
        if t.type == 'op' and t.value in ('-', '+', '!', '~'):
            self.next()
            operand = self.parse_unary()
            if t.value == '+':
                return operand
            if t.value == '-' and operand.kind == 'num':
                return Node('num', t.line, value=-operand.value, type=operand.type)
            return Node('unop', t.line, op=t.value, x=operand)
        if t.type == 'op' and t.value in ('++', '--'):
            self.next()
            operand = self.parse_unary()
            return Node('incdec', t.line, op=t.value, target=operand, post=False)
        # C style cast: (float)x
        if self.is_op('(') and self.is_type(self.peek()) and self.is_op(')', self.peek(2)):
            self.next()
            typ = self.next().value
            self.next()
            operand = self.parse_unary()
            return Node('cast', t.line, type=typ, x=operand)
        return self.parse_postfix()

    def parse_postfix(self):
        e = self.parse_primary()
        while True:
            t = self.tok
            if self.is_op('['):
                self.next()
                idx = self.parse_expr()
                self.expect(']')
                e = Node('index', t.line, x=e, i=idx)
            elif self.is_op('.'):
                self.next()
                name = self.expect_ident()
                e = Node('member', t.line, x=e, name=name)
            elif t.type == 'op' and t.value in ('++', '--'):
                self.next()
                e = Node('incdec', t.line, op=t.value, target=e, post=True)
            else:
                return e

    def parse_args(self):
        args = []
        self.expect('(')
        while not self.is_op(')'):
            args.append(self.parse_assign())
            if not self.accept(','):
                break
        self.expect(')')
        return args

    def parse_primary(self):
        t = self.tok
        if t.type == 'int':
            self.next()
            return Node('num', t.line, value=t.value, type='int')
        if t.type == 'float':
            self.next()
            return Node('num', t.line, value=t.value, type='float')
        if t.type == 'string':
            self.next()
            s = t.value
            # Adjacent string literals concatenate.
            while self.tok.type == 'string':
                s += self.next().value
            return Node('str', t.line, value=s)
        if t.type == 'attr':
            self.next()
            prefix, is_arr, name = t.value
            typ = ATTR_PREFIX.get(prefix) if prefix else None
            if typ and is_arr:
                typ += '[]'
            prev = self.bindings.get(name)
            if typ is not None or prev is None:
                self.bindings[name] = typ or prev
            return Node('attr', t.line, name=name, type=typ)
        if self.is_op('('):
            self.next()
            e = self.parse_expr_list()
            self.expect(')')
            return e
        if self.is_op('{'):
            self.next()
            items = []
            while not self.is_op('}'):
                items.append(self.parse_assign())
                if not self.accept(','):
                    break
            self.expect('}')
            return Node('brace', t.line, items=items)
        if t.type == 'ident':
            self.next()
            name = t.value
            if name in TYPE_NAMES:
                # Constructor / function style cast: vector(1), float(x), vector[](...)
                if self.is_op('[') and self.is_op(']', self.peek()):
                    self.pos += 2
                    name += '[]'
                if self.is_op('('):
                    args = self.parse_args()
                    return Node('cast', t.line, type=name, x=args[0] if len(args) == 1 else None, args=args)
                raise VexSyntaxError("Unexpected type name '%s'" % name, t.line, t.col)
            if self.is_op('('):
                args = self.parse_args()
                return Node('call', t.line, name=name, args=args)
            return Node('var', t.line, name=name)
        raise VexSyntaxError("Unexpected %s" % (repr(t.value) if t.value is not None else 'end of code'), t.line, t.col)


def parse(src):
    return Parser(src).parse_program()
