import re

formulas = [
    r"""\operatorname{dir}(A) =
\begin{cases}
+1, & f_A(1) - f_A(0) > \varepsilon, \\
-1, & f_A(1) - f_A(0) < -\varepsilon, \\
\ \ 0, & \text{w przeciwnym razie (singleton)},
\end{cases}
\qquad \varepsilon = 10^{-12}""",
    r"""\mu_A(x) =
\begin{cases}
1, & x \in [\min(f_A(1), g_A(1)),\ \max(f_A(1), g_A(1))], \\[2pt]
\max\big(f_A^{-1}(x),\, g_A^{-1}(x)\big), & \text{poza jądrem, wewnątrz nośnika}, \\[2pt]
0, & \text{poza nośnikiem}.
\end{cases}"""
]

for idx, formula in enumerate(formulas):
    print(f"--- Formula {idx+1} ---")
    start_idx = formula.find(r'\begin{cases}')
    end_idx = formula.find(r'\end{cases}')
    lhs = formula[:start_idx].strip()
    cases_content = formula[start_idx + len(r'\begin{cases}'):end_idx].strip()
    trailing_math = formula[end_idx + len(r'\end{cases}'):].strip()
    
    raw_lines = re.split(r'\\\\(?:\[.*?\])?', cases_content)
    cases_val = []
    cases_cond = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('&')
        if len(parts) == 2:
            cases_val.append(parts[0].strip())
            cases_cond.append(parts[1].strip())
        else:
            cases_val.append(line)
            cases_cond.append("")
            
    if trailing_math:
        if cases_cond:
            cases_cond[-1] = cases_cond[-1] + r'\quad\quad ' + trailing_math
            
    print("LHS:", repr(lhs))
    print("Values:", cases_val)
    print("Conditions:", cases_cond)
