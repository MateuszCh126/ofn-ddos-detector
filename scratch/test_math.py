import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for clean math rendering
rcParams['mathtext.fontset'] = 'dejavusans'

def latex_to_png(latex_str, filepath, dpi=300):
    if not latex_str.startswith('$'):
        latex_str = f"${latex_str}$"
        
    # Create figure with transparent background
    fig = plt.figure(figsize=(0.1, 0.1), facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Render text
    t = ax.text(0.5, 0.5, latex_str, fontsize=14, ha='center', va='center', color='#2D3748')
    
    # Draw to get bbox
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = t.get_window_extent(renderer)
    
    # Convert bbox to inches and add padding
    width_in = (bbox.width + 30) / dpi
    height_in = (bbox.height + 20) / dpi
    fig.set_size_inches(width_in, height_in)
    
    # Save image with transparent background
    plt.savefig(filepath, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

if __name__ == '__main__':
    os.makedirs('scratch/temp_math', exist_ok=True)
    latex_to_png(r'A = (f_A,\, g_A), \qquad f_A, g_A : [0,1] \to \mathbb{R}', 'scratch/temp_math/formula_1.png')
    print("Done! Image saved to scratch/temp_math/formula_1.png")
