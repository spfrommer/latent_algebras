from __future__ import annotations
from collections import OrderedDict, defaultdict

import contextlib
from dataclasses import dataclass, asdict
import itertools
import random
import click
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import math
import functools
import tabulate

import boolean
from boolean import Expression, Symbol
from boolean.boolean import BaseElement, _TRUE, _FALSE

from jaxtyping import Float, Bool, jaxtyped
from beartype import beartype as typechecker
from einops import rearrange, reduce, repeat, pack
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from latalg.utils import model_utils
from latalg.utils import torch_utils as TU


# ------------------------------------------------------------------------------------ #
#                         Expression generation and evaluation                         #
# ------------------------------------------------------------------------------------ #
    
parser = boolean.BooleanAlgebra()

def symbolize(values: List[Tensor]) -> Dict[str, Tensor]:
    symbol = {i: chr(i + ord('a')) for i in range(len(values))}
    return {symbol[i]: values[i] for i in range(len(values))}


@jaxtyped
@typechecker
def evaluate_expression(
        expr: boolean.Expression,
        symbols: Union[Dict[str, Tensor], List[Tensor]],
        ba: SetAlgebra,
    ) -> Tensor:

    if isinstance(symbols, list):
        symbols = symbolize(symbols)

    if isinstance(expr, Symbol):
        assert expr.obj in symbols, f'Symbol {expr.obj} not in vars'
        return symbols[expr.obj]
    elif isinstance(expr, BaseElement):
        raise ValueError(f'Invalid boolean value: {expr}')
    else:
        args_evaled = [evaluate_expression(arg, symbols, ba) for arg in expr.args]
        ops = {boolean.AND: ba.band, boolean.OR: ba.bor}
        return ops[type(expr)](*args_evaled)

def random_expression(literal_n: int) -> Expression:
    literals = [chr(ord('a') + i) for i in range(literal_n)]
    expressions = list(parser.symbols(*literals))
    _, _, NOT, AND, OR, _ = parser.definition()
    while len(expressions) > 1:
        operation = random.choice([AND, OR])
        arg_n = {AND: 2, OR: 2}[operation]
        index_expr = random.sample(list(enumerate(expressions)), arg_n)
        indices, exprs = zip(*index_expr)
        new_expression = operation(*exprs)
        
        for index in sorted(indices, reverse=True):
            del expressions[index]

        expressions.append(new_expression)

    return expressions[0]

class Law():
    def is_applicable(self, expr: Expression) -> bool:
        raise NotImplementedError
    
    def apply(self, expr: Expression, random_symbol: Symbol) -> Expression:
        raise NotImplementedError

class CommutativityLaw(Law):
    def is_applicable(self, expr: Expression) -> bool:
        return True
    
    def apply(self, expr: Expression, _) -> Expression:
        args = list(expr.args)
        args.reverse()
        expr.args = tuple(args)
        return expr
        
class AssociativityLaw(Law):
    def is_applicable(self, expr: Expression) -> bool:
        op = type(expr)
        return isinstance(expr.args[0], op) or isinstance(expr.args[1], op)
    
    def apply(self, expr: Expression, _) -> Expression:
        check_order = random.sample([0, 1], 2)
        for i in check_order:
            if isinstance(expr.args[i], type(expr)):
                other_i = 1 - i
                new_args = [None, None]
                new_args[i] = expr.args[i].args[0]
                new_args[other_i] = type(expr)(expr.args[i].args[1], expr.args[other_i])
                return type(expr)(*new_args)

class AbsorptionLaw(Law):
    def is_applicable(self, expr: Expression) -> bool:
        op, dual_op = type(expr), expr.dual
        if isinstance(expr.args[0], Symbol):
            if isinstance(expr.args[1], dual_op):
                if expr.args[0] == expr.args[1].args[0]:
                    # "Simplification"
                    return True
        # Randomly complexify one of the symbols
        return isinstance(expr.args[0], Symbol) or isinstance(expr.args[1], Symbol)
    
    def apply(self, expr: Expression, random_symbol: Symbol) -> Expression:
        op, dual_op = type(expr), expr.dual
        if isinstance(expr.args[0], Symbol) and isinstance(expr.args[1], dual_op):
            if expr.args[0] == expr.args[1].args[0]:
                return expr.args[0]

        check_order = random.sample([0, 1], 2)
        random_op, other_op = random.sample([op, dual_op], 2)
        for i in check_order:
            if isinstance(expr.args[i], Symbol):
                other_i = 1 - i
                new_args = [None, None]
                new_args[other_i] = expr.args[other_i]
                new_args[i] = random_op(
                    expr.args[i], other_op(expr.args[i], random_symbol)
                )
        
                return op(*new_args)
    
class DistributivityLaw(Law):
    def is_applicable(self, expr: Expression) -> bool:
        op, dual_op = type(expr), expr.dual
        return isinstance(expr.args[1], dual_op)
    
    def apply(self, expr: Expression, _) -> Expression:
        op, dual_op = type(expr), expr.dual
        # "Forward" law application
        new_args = [None, None]
        new_args[0] = op(expr.args[0], expr.args[1].args[0])
        new_args[1] = op(expr.args[0], expr.args[1].args[1])
        return dual_op(*new_args)


all_laws = [
    CommutativityLaw(), AssociativityLaw(), AbsorptionLaw(), DistributivityLaw()
]

def random_equivalent_expression(expr: Expression, law_n: int) -> Expression:
    """Simplify and randomly apply laws to an expression."""
    def random_desired_laws():
        return random.sample(all_laws, 4)
    
    def extract_exprs(expr: Expression) -> List[Tuple[Expression, Expression]]:
        all_expressions = []
        def inner(expr, parent, arg_index):
            if isinstance(expr, boolean.Symbol):
                return

            if parent is not None:
                all_expressions.append((expr, parent, arg_index))
            for i, arg in enumerate(expr.args):
                inner(arg, expr, i)
                
        inner(expr, None, None)
        return all_expressions

    expr = boolean.NOT(expr) # Wrapper
    for _ in range(law_n):
        desired_laws = random_desired_laws()
        expressions = extract_exprs(expr)
        expressions = random.sample(expressions, len(expressions))
        
        for law in desired_laws:
            for inner_expr, parent, arg_index_in_parent in expressions:
                if not law.is_applicable(inner_expr):
                    continue
                
                random_symbol = random.choice(list(expr.symbols))
                new_expr = law.apply(inner_expr, random_symbol)
                assert new_expr is not None
                
                # import pdb; pdb.set_trace()
                parent_args = list(parent.args)
                parent_args[arg_index_in_parent] = new_expr
                parent.args = tuple(parent_args)
                break
            else:
                continue
            break
    
    return expr.args[0]


def check_expressions_equal(expr1: Expression, expr2: Expression) -> bool:
    _, _, NOT, AND, OR, _ = parser.definition()
    symbols = list(expr1.symbols)
    all_assignments = list(itertools.product([True, False], repeat=len(symbols)))
    
    def eval(expr: Expression, assignment):
        if isinstance(expr, Symbol):
            return assignment[expr]
        elif isinstance(expr, OR):
            return eval(expr.args[0], assignment) or eval(expr.args[1], assignment)
        elif isinstance(expr, AND):
            return eval(expr.args[0], assignment) and eval(expr.args[1], assignment) 
        raise ValueError(f'Invalid expression: {expr}')
    
    for assignment in all_assignments:
        assignment_dict = {symbols[i]: assignment[i] for i in range(len(symbols))}
        
        eval1 = eval(expr1, assignment_dict)
        eval2 = eval(expr2, assignment_dict)
        
        if eval1 != eval2:
            return False
    return True


# if __name__ == "__main__":
#     for _ in range(10):
#         randexpr = random_expression(10)
#         mutatedexpr = random_equivalent_expression(randexpr, 5)
        
#         assert check_expressions_equal(randexpr, mutatedexpr)
#     import sys
#     sys.exit()

# ------------------------------------------------------------------------------------ #
#                                Set Algebra Definitions                               #
# ------------------------------------------------------------------------------------ #

@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class SetAlgebra:
    def forward(self, *args: Tensor, operation: str) -> Tensor:
        ops = {'and': self.band, 'or': self.bor}
        return ops[operation](*args)
    
    def list_forward(self, args: List[Tensor], operation: str) -> Tensor:
        ops = {'and': self.band, 'or': self.bor}
        return functools.reduce(ops[operation], args)

    def band(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
    
    def bor(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
    

@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class InducedAlgebra(SetAlgebra):
    def __init__(self, base_algebra: SetAlgebra, phi, phi_inv):
        self.base_algebra = base_algebra
        self.phi = phi
        self.phi_inv = phi_inv

    def band(self, x: Tensor, y: Tensor) -> Tensor:
        if len(x.shape) > 1:
            return self.phi_inv(self.base_algebra.band(self.phi(x), self.phi(y)))
        phi_x = self.phi(x.unsqueeze(0)).squeeze(0)
        phi_y = self.phi(y.unsqueeze(0)).squeeze(0)
        base_out = self.base_algebra.band(phi_x, phi_y).unsqueeze(0)
        return self.phi_inv(base_out).squeeze(0)
    
    def bor(self, x: Tensor, y: Tensor) -> Tensor:
        if len(x.shape) > 1:
            return self.phi_inv(self.base_algebra.bor(self.phi(x), self.phi(y)))
        phi_x = self.phi(x.unsqueeze(0)).squeeze(0)
        phi_y = self.phi(y.unsqueeze(0)).squeeze(0)
        base_out = self.base_algebra.bor(phi_x, phi_y).unsqueeze(0)
        return self.phi_inv(base_out).squeeze(0)


@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class SetMembershipAlgebra(SetAlgebra):
    # Here, the "representation" of a set is the value of the predictor
    # on some collection of random points
    def band(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.minimum(x, y)
    
    def bor(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.maximum(x, y)



class DirectParamOperation(nn.Module):
    """A directly parameterized operation on set representations."""
    def __init__(
            self,
            arg_n: int,
            input_d: int,
            intermediate_d: int,
            mlp_params: model_utils.MLPParams,
            permutation_invariant=True,
            **kwargs
        ):

        super().__init__(**kwargs)

        self.arg_n = arg_n
        self.permutation_invariant = permutation_invariant
        self.input_d = input_d
        self.intermediate_d = intermediate_d

        self.intermediate_net = model_utils.create_mlp_from_params(
            self.input_d, self.intermediate_d, mlp_params,
        )

        combination_in = (
            intermediate_d if permutation_invariant else intermediate_d * arg_n
        )
        self.combination_net = model_utils.create_mlp_from_params(
            combination_in, self.input_d, mlp_params
        )

    @typechecker
    @jaxtyped
    def forward(self, *args: Tensor) -> Tensor:
        inters = torch.stack([self.intermediate_net(arg) for arg in args], dim=0)
        if self.permutation_invariant:
            inters = reduce(inters, 'arg b latent -> b latent', 'sum')
        else:
            inters = rearrange(inters, 'arg b latent -> b (arg latent)')
        return self.combination_net(inters)


@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class DirectParamLatentAlgebra(nn.Module, SetAlgebra):
    def __init__(
            self,
            embedding_d: int,
            mlp_params: model_utils.MLPParams,
            intermediate_d=256,
            permutation_invariant=True,
        ):

        super().__init__()

        args = {
            'input_d': embedding_d,
            'mlp_params': mlp_params,
            'intermediate_d': intermediate_d,
            'permutation_invariant': permutation_invariant,
        }

        self.band_op = DirectParamOperation(arg_n=2, **args)
        self.bor_op = DirectParamOperation(arg_n=2, **args)

    def forward(self, *args: Tensor, operation: str) -> Tensor:
        # Need to override Module definition of forward
        return SetAlgebra.forward(self, *args, operation=operation)

    def band(self, x: Tensor, y: Tensor) -> Tensor:
        if len(x.shape) == 1:
            return self.band_op(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        return self.band_op(x, y)
    
    def bor(self, x: Tensor, y: Tensor) -> Tensor:
        if len(x.shape) == 1:
            return self.bor_op(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
        return self.bor_op(x, y)
    

# ------------------------------------------------------------------------------------ #
#                               Algebra Option Generation                              #
# ------------------------------------------------------------------------------------ #

def cyclic_add(a: Tensor, b: Tensor) -> Tensor:
    assert (len(a.shape) == 1) and (len(b.shape) == 1)
    return torch.roll(a, shifts=1, dims=0) + b

def matrix_mul(a: Tensor, b: Tensor) -> Tensor:
    assert (len(a.shape) == 1) and (len(b.shape) == 1)
    sqrt_size = int(math.sqrt(a.shape[0]))
    a_matrix = rearrange(a, '(s1 s2) -> s1 s2', s1=sqrt_size)
    b_matrix = rearrange(b, '(s1 s2) -> s1 s2', s1=sqrt_size)
    ab = torch.matmul(a_matrix, b_matrix)
    return rearrange(ab, 's1 s2 -> (s1 s2)')

def scale_add(a: Tensor, b: Tensor) -> Tensor:
    assert (len(a.shape) == 1) and (len(b.shape) == 1)
    return 2 * a + 2 * b

# If add/remove, update the run script
operations = OrderedDict([
    ('max', torch.max),
    ('min', torch.min),
    ('add', torch.add),
    ('sub', torch.sub),
    ('mul', torch.mul),
    ('scale_add', scale_add), # Commutative but not associative
    ('matrix_mul', matrix_mul), # Associative but not commutative 
    ('cyclic_add', cyclic_add), # Neither commutative nor associative
])

properties = OrderedDict([
    (('commutativity', False), 'a & b = b & a'),
    (('commutativity', True), 'a | b = b | a'),
    (('associativity', False), 'a & (b & c) = (a & b) & c'),
    (('associativity', True), 'a | (b | c) = (a | b) | c'),
    (('absorption', False), 'a | (a & b) = a'),
    (('absorption', True), 'a & (a | b) = a'),
    (('distributivity', False), 'a | (b & c) = (a | b) & (a | c)'),
    (('distributivity', True), 'a & (b | c) = (a & b) | (a & c)'),
])

    

@TU.for_all_methods(jaxtyped)
@TU.for_all_methods(typechecker)
class TransportedAlgebra(SetAlgebra):
    def __init__(self, and_operation: str, or_operation: str):
        self.and_operation = and_operation
        self.or_operation = or_operation
        self.and_function = operations[and_operation]
        self.or_function = operations[or_operation]

    def band(self, x: Tensor, y: Tensor) -> Tensor:
        return self.and_function(x, y)
    
    def bor(self, x: Tensor, y: Tensor) -> Tensor:
        return self.or_function(x, y)
    
    def __str__(self):
        return f'and: {self.and_operation} || or: {self.or_operation}'


transported_algebra_ops = []
transported_algebras = []
for i, op1 in enumerate(list(operations.items())):
    for j, op2 in enumerate(list(operations.items())[i+1:]):
        transported_algebra_ops.append((op1[0], op2[0]))
        transported_algebras.append(TransportedAlgebra(op1[0], op2[0]))
        

def get_transported_algebra_index(transported_index: int) -> TransportedAlgebra:
    return transported_algebras[transported_index]

def get_transported_algebra_ops(op1: str, op2: str) -> TransportedAlgebra:
    return TransportedAlgebra(op1, op2)

def num_tranported_combinations() -> int:
    return len(transported_algebras)

def get_property_types() -> List[str]:
    property_types_repeated = [property[0] for property in properties.keys()]
    unique = list(OrderedDict.fromkeys(property_types_repeated))
    return unique


def test_property(
        algebra: SetAlgebra,
        property: str,
        property_prime: bool, # (https://www.jstor.org/stable/43672165)
    ) -> bool:
    
    in_d = 1024 # So that it also works for DirectParamLatentAlgebra
    dummy_a, dummy_b, dummy_c = torch.rand(in_d), torch.rand(in_d), torch.rand(in_d)
    if isinstance(algebra, DirectParamLatentAlgebra):
        dummy_a = dummy_a.cuda()
        dummy_b = dummy_b.cuda()
        dummy_c = dummy_c.cuda()
    symbols = symbolize([dummy_a, dummy_b, dummy_c])
    
    test_equality = properties[(property, property_prime)]
    
    lhs, rhs = test_equality.split('=')
    
    lhs_expr, rhs_expr = parser.parse(lhs), parser.parse(rhs)
    
    lhs_eval = evaluate_expression(lhs_expr, symbols, algebra)
    rhs_eval = evaluate_expression(rhs_expr, symbols, algebra)
    
    return torch.allclose(lhs_eval, rhs_eval)
    
def num_satisfied_properties(algebra: SetAlgebra) -> int:
    satisfied = 0
    for property in properties:
        if test_property(algebra, *property):
            satisfied += 1
    return satisfied  
    
def num_satisfied_for_property_type(algebra: SetAlgebra) -> Dict[str, int]:
    # Returns {'commutativity': 1, ...}, each between 0 and 2 inclusive
    property_types = get_property_types()
    
    satisfied = defaultdict(int)
    for property_type in property_types:
        for property_prime in [True, False]:
            if test_property(algebra, property_type, property_prime):
                satisfied[property_type] += 1
    
    return satisfied

def print_properties():
    print_latex = True

    if print_latex:
        operation_lookup = {
            'max': '\\max',
            'min': '\\min',
            'add': '+',
            'sub': '-',
            'mul': '\\odot',
            'scale_add': '+_s',
            'matrix_mul': '\\times_{\\text{mat}}',
            'cyclic_add': '+_c',
        }
        and_sym, or_sym, to_sym = '\\cap^{\M}', '\\cup^{\M}', '='
        check_sym, x_sym = '\\cmark', '\\xmark'
        count_sym = '\\#'
    else:
        operation_lookup = {
            'max': 'max',
            'min': 'min',
            'add': '+',
            'sub': '-',
            'mul': '×',
            'scale_add': '+s',
            'matrix_mul': '×mat',
            'cyclic_add': '+c',
        }
        and_sym, or_sym, to_sym = '∧', '∨', '→'
        check_sym, x_sym = '✔', '✘'
        count_sym = '#'

    
    def algebra_str(algebra):
        and_op_sym = operation_lookup[algebra.and_operation]
        or_op_sym = operation_lookup[algebra.or_operation]
        and_string = f'{and_sym} {to_sym} {and_op_sym}'
        or_string = f'{or_sym} {to_sym} {or_op_sym}'
        and_string = f'${and_string}$' if print_latex else and_string
        or_string = f'${or_string}$' if print_latex else or_string
        return and_string, or_string
    
    rows = []
    for algebra in transported_algebras:
        row = [*algebra_str(algebra), num_satisfied_properties(algebra)]
        for property in properties:
            if property[1] == True:
                continue
            symbols = {True: check_sym, False: x_sym}
            sat = symbols[test_property(algebra, property[0], False)] 
            perm_sat = symbols[test_property(algebra, property[0], True)]
            row.append(f'{sat} {perm_sat}')
        rows.append(row)
    
    headers = ['Operations', '', count_sym]
    for property in properties:
        if property[1] == True:
            continue
        headers.append(property[0].capitalize() + " ($^*$)")
        
    # Sort rows by the second column (total number of satisfied properties)
    rows = sorted(rows, key=lambda x: x[2], reverse=True)
    
    fmt = 'latex_raw' if print_latex else 'simple'
    table_str = tabulate.tabulate(rows, headers=headers, tablefmt=fmt)
    
    if print_latex:
        # center align columns
        import re
        pattern = f'l{{4,}}'
        table_str = re.sub(pattern, lambda match: 'c' * len(match.group()), table_str)

    import pyperclip
    print(table_str)
    pyperclip.copy(table_str)


if __name__ == "__main__":
    print_properties()

