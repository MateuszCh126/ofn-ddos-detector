import re

formula = r"\boxed{\,A - A = (0, 0) = \mathbf{0}\,}"

boxed_match = re.match(r'\\boxed\{(.*?)\}', formula.strip())
if boxed_match:
    inner_formula = boxed_match.group(1).strip()
    print("Group 1:", repr(inner_formula))
    if inner_formula.startswith(r'\,'):
        inner_formula = inner_formula[2:]
    print("After startswith slice:", repr(inner_formula))
    if inner_formula.endswith(r'\,'):
        inner_formula = inner_formula[:-2]
    print("After endswith slice:", repr(inner_formula))
