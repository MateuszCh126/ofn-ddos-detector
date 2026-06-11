import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

formula = r"\text{poza jądrem, wewnątrz nośnika}"

fig = plt.figure(figsize=(6, 1))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
try:
    ax.text(0.5, 0.5, f"${formula}$", fontsize=14, ha='center', va='center')
    plt.savefig('scratch/temp_math/polish_test.png', dpi=300, bbox_inches='tight')
    print("Success rendering Polish characters in math mode!")
except Exception as e:
    print("Failed rendering Polish characters in math mode:", e)
finally:
    plt.close(fig)
