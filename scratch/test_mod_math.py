import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

formulas = {
    5: r"A - A = (0, 0) = \mathbf{0}",
    18: r"\gamma = (w_1, \dots, w_R, \theta_a, \rho, \phi, k_a, k_c)"
}

os.makedirs('scratch/temp_math', exist_ok=True)

for num, formula in formulas.items():
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    try:
        ax.text(0.5, 0.5, f"${formula}$", fontsize=12, ha='center', va='center')
        plt.savefig(f'scratch/temp_math/formula_{num}.png', dpi=300, bbox_inches='tight')
        print(f"Success formula {num}")
    except Exception as e:
        print(f"FAIL formula {num}: {e}")
    finally:
        plt.close(fig)
