import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

# Test matrix in mathtext
# Matplotlib mathtext matrix syntax: \matrix{ a & b \cr c & d }
matrix_formula = r"\operatorname{dir}(A) = \left\{ \matrix{ +1, & f_A(1) - f_A(0) > \varepsilon \cr -1, & f_A(1) - f_A(0) < -\varepsilon \cr 0, & \text{w przeciwnym razie (singleton)} } \right."

fig = plt.figure(figsize=(8, 2))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
try:
    ax.text(0.5, 0.5, f"${matrix_formula}$", fontsize=12, ha='center', va='center')
    plt.savefig('scratch/temp_math/matrix_test.png', dpi=300, bbox_inches='tight')
    print("Success rendering matrix!")
except Exception as e:
    print("Failed rendering matrix:", e)
finally:
    plt.close(fig)
