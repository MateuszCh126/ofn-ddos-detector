import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

formula_with_newlines = r"""\mu_A(x) =
\begin{cases}
1, & x \in [\min(f_A(1), g_A(1)),\ \max(f_A(1), g_A(1))], \\[2pt]
\max\big(f_A^{-1}(x),\, g_A^{-1}(x)\big), & \text{poza jądrem, wewnątrz nośnika}, \\[2pt]
0, & \text{poza nośnikiem}.
\end{cases}"""

# 1. Try with newlines
fig1 = plt.figure(figsize=(6, 2))
ax1 = fig1.add_axes([0, 0, 1, 1])
ax1.axis('off')
try:
    ax1.text(0.5, 0.5, f"${formula_with_newlines}$", fontsize=12, ha='center', va='center')
    plt.savefig('scratch/temp_math/with_newlines.png', dpi=300, bbox_inches='tight')
    print("Success with newlines")
except Exception as e:
    print("Failed with newlines:", e)
finally:
    plt.close(fig1)

# 2. Try without newlines (replacing them with space)
formula_no_newlines = formula_with_newlines.replace('\n', ' ')
fig2 = plt.figure(figsize=(6, 2))
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.axis('off')
try:
    ax2.text(0.5, 0.5, f"${formula_no_newlines}$", fontsize=12, ha='center', va='center')
    plt.savefig('scratch/temp_math/without_newlines.png', dpi=300, bbox_inches='tight')
    print("Success without newlines")
except Exception as e:
    print("Failed without newlines:", e)
finally:
    plt.close(fig2)
