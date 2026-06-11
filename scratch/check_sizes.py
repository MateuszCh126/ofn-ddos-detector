import os
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

formulas = [
    r"A = (f_A,\, g_A), \qquad f_A, g_A : [0,1] \to \mathbb{R}",
    r"""\operatorname{dir}(A) =
\begin{cases}
+1, & f_A(1) - f_A(0) > \varepsilon, \\
-1, & f_A(1) - f_A(0) < -\varepsilon, \\
\ \ 0, & \text{w przeciwnym razie (singleton)},
\end{cases}
\qquad \varepsilon = 10^{-12}""",
    r"""A + B = (f_A + f_B,\; g_A + g_B), \qquad
A - B = (f_A - f_B,\; g_A - g_B)""",
    r"""c \cdot A = (c f_A,\; c g_A) \ \ \text{dla } c \in \mathbb{R}, \qquad
-A = (-f_A,\; -g_A), \qquad
A \cdot B = (f_A f_B,\; g_A g_B)""",
    r"A - A = (0, 0) = \mathbf{0}",
    r"""\mu_A(x) =
\begin{cases}
1, & x \in [\min(f_A(1), g_A(1)),\ \max(f_A(1), g_A(1))], \\[2pt]
\max\big(f_A^{-1}(x),\, g_A^{-1}(x)\big), & \text{poza jądrem, wewnątrz nośnika}, \\[2pt]
0, & \text{poza nośnikiem}.
\end{cases}""",
    r"\operatorname{COG}(A) = \frac{\int x\, \mu_A(x)\, dx}{\int \mu_A(x)\, dx}",
    r"\operatorname{MOA}(A) = \frac{1}{2}\left(\int_0^1 f_A(y)\, dy + \int_0^1 g_A(y)\, dy\right)",
    r"f(y) = a + y(b - a), \qquad g(y) = c + (1-y)(d - c)",
    r"""m_r = \operatorname{med}(H), \qquad
s_r = \max\big(1.4826 \cdot \operatorname{med}\lvert H - m_r \rvert,\ s_{\min}\big)""",
    r"z_r(t) = \operatorname{clip}\!\left(\frac{x_r(t) - m_r}{s_r},\, -8,\, 8\right)",
    r"""z^{(c)}_r(t) = \frac{\sum_j v_j\, z_{r,j}(t)}{\sum_j v_j},
\qquad
u^{(c)}_r(t) = \frac{\sum_j v_j\, \max(z_{r,j}(t), 0)}{\sum_j v_j}""",
    r"""\hat\beta = \frac{\sum_{i=1}^{W} (i - \bar i)(z_i - \bar z)}{\sum_{i=1}^{W} (i - \bar i)^2},
\qquad
T = \hat\beta \cdot (W - 1)""",
    r"""d_r = \operatorname{sign}(T) \cdot \mathbb{1}\big[\lvert T \rvert > \varepsilon_T\big],
\qquad \varepsilon_T = 2{,}2""",
    r"""G = \frac{1}{\Omega} \left(
\sum_{r:\, d_r > 0} w_r A_r
\;-\; \sum_{r:\, d_r < 0} w_r A_r
\;+\; \kappa \sum_{r:\, d_r = 0} w_r A_r
\right),
\qquad
\Omega = \sum_{r} w_r^{\text{eff}}""",
    r"""S(t) = \max\big(\operatorname{MOA}(G),\, 0\big)
= \max\!\left( \frac{1}{\Omega}\sum_r \pm\, w_r^{\text{eff}}\operatorname{MOA}(A_r),\ 0 \right)""",
    r"""\text{recall} = \frac{TP}{TP + FN}, \quad
\text{precision} = \frac{TP}{TP + FP}, \quad
F_1 = \frac{2PR}{P + R}, \quad
\text{FPR} = \frac{FP}{FP + TN}""",
    r"\gamma = (w_1, \dots, w_R, \theta_a, \rho, \phi, k_a, k_c)",
    r"""J(\gamma) = 0{,}55\,(1 - \text{recall}) + 0{,}30\,\text{FPR}
+ 0{,}15\,\frac{\text{delay}}{|T| - t_0}""",
    r"""\delta_t = V(t) - \hat\mu_{t-1}, \qquad
\hat\mu_t = \hat\mu_{t-1} + \alpha\,\delta_t, \qquad
\hat\sigma^2_t = (1 - \alpha)\big(\hat\sigma^2_{t-1} + \alpha\,\delta_t^2\big)"""
]

for idx, formula in enumerate(formulas):
    latex_str = formula
    if not latex_str.startswith('$'):
        latex_str = f"${latex_str}$"
        
    fig = plt.figure(figsize=(0.1, 0.1), facecolor='none')
    fig.set_dpi(100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    t = ax.text(0.5, 0.5, latex_str, fontsize=12.5, ha='center', va='center', color='#2D3748')
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = t.get_window_extent(renderer)
    
    # Let's print sizes
    w_px = bbox.width
    h_px = bbox.height
    
    # Old calculation
    old_w_in = (w_px + 40) / 300.0
    
    # New calculation
    new_w_in = (w_px + 40) / 100.0
    
    print(f"Formula {idx+1}: px_width={w_px:.1f}, old_w_in={old_w_in:.3f} in, new_w_in={new_w_in:.3f} in")
    plt.close(fig)
