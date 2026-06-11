import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'dejavusans'

def render_cases_custom(lhs, cases_list, filepath, dpi=300):
    # Create figure
    fig = plt.figure(figsize=(8, 2.2), facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 1. Render Left-Hand Side (LHS)
    t_lhs = ax.text(0.05, 0.5, f"${lhs}$", fontsize=15, ha='left', va='center', color='#2D3748')
    
    # Draw to find LHS bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_lhs = t_lhs.get_window_extent(renderer)
    
    # 2. Render large brace next to LHS
    # We convert LHS right coordinate to figure fraction
    # Get figure width in pixels
    fig_width_px = fig.bbox.width
    lhs_right_frac = bbox_lhs.x1 / fig_width_px
    
    # Place brace slightly to the right of LHS
    brace_x = lhs_right_frac + 0.015
    t_brace = ax.text(brace_x, 0.5, r"$\{$", fontsize=48, ha='left', va='center', color='#2D3748', weight='light')
    
    # Draw to find brace bounding box
    fig.canvas.draw()
    bbox_brace = t_brace.get_window_extent(renderer)
    brace_right_frac = bbox_brace.x1 / fig_width_px
    
    # 3. Render case lines next to brace
    cases_x = brace_right_frac + 0.015
    
    # We have 3 cases. Let's stack them at y = 0.78, 0.50, 0.22
    y_coords = [0.78, 0.50, 0.22]
    t_cases = []
    for val_str, cond_str, y_coord in zip(cases_list[0], cases_list[1], y_coords):
        case_text = f"${val_str} \\quad {cond_str}$"
        t_case = ax.text(cases_x, y_coord, case_text, fontsize=14, ha='left', va='center', color='#2D3748')
        t_cases.append(t_case)
        
    # Redraw to get final dimensions
    fig.canvas.draw()
    
    # Find the maximum right coordinate of all case lines
    max_x1 = 0
    for t_case in t_cases:
        bbox = t_case.get_window_extent(renderer)
        if bbox.x1 > max_x1:
            max_x1 = bbox.x1
            
    # Calculate required figure width in inches
    dpi_f = float(dpi)
    draw_dpi = 100.0
    
    # Max x1 in pixels to inches, plus padding
    width_in = (max_x1 + 40) / draw_dpi
    # Set height based on 3 cases (around 1.4 inches)
    height_in = 1.6
    
    fig.set_size_inches(width_in, height_in)
    
    # Update coordinates to fit new figure width
    # Since we used coordinate fractions, Matplotlib will scale them, but let's make sure
    # they are positioned correctly. We'll use absolute coordinates or redraw.
    # To keep it simple, we can just save it with bbox_inches='tight'!
    plt.savefig(filepath, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return width_in

if __name__ == '__main__':
    os.makedirs('scratch/temp_math', exist_ok=True)
    
    # Formula 2
    lhs2 = r"\operatorname{dir}(A) ="
    cases_list2 = (
        [r"+1,", r"-1,", r"\ \ 0,"],
        [r"f_A(1) - f_A(0) > \varepsilon,", r"f_A(1) - f_A(0) < -\varepsilon,", r"\text{w przeciwnym razie (singleton)}"]
    )
    w = render_cases_custom(lhs2, cases_list2, 'scratch/temp_math/formula_2_custom.png')
    print("Done! Width:", w)
