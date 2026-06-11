import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

def try_render(latex_str, filename):
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    try:
        ax.text(0.5, 0.5, f"${latex_str}$", fontsize=12, ha='center', va='center')
        plt.savefig(f'scratch/temp_math/{filename}.png', dpi=300, bbox_inches='tight')
        print(f"Successfully rendered: {filename}")
    except Exception as e:
        print(f"Failed to render {filename}: {e}")
    finally:
        plt.close(fig)

if __name__ == '__main__':
    os.makedirs('scratch/temp_math', exist_ok=True)
    
    # Try case block
    cases_test = r'''\operatorname{dir}(A) =
\begin{cases}
+1, & f_A(1) - f_A(0) > \varepsilon, \\
-1, & f_A(1) - f_A(0) < -\varepsilon, \\
\ \ 0, & \text{w przeciwnym razie (singleton)},
\end{cases}'''
    try_render(cases_test, 'cases')
