import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math

# Dictionary for rebar sizes (diameter in inches)
rebar_diameters = {
    '#3': 0.375,
    '#4': 0.5,
    '#5': 0.625,
    '#6': 0.75,
    '#7': 0.875,
    '#8': 1.0,
    '#9': 1.128,
    '#10': 1.27,
    '#11': 1.41,
    '#14': 1.693,
    '#18': 2.257
}

# Function to calculate area of one bar
def bar_area(bar_size):
    d = rebar_diameters[bar_size]
    return math.pi * (d / 2) ** 2

# Function to check validity: spacing, covers, ratios
def check_validity(b, h, fc, fy, layers, side_cover, bottom_cover, top_cover):
    warnings = []
    
    if not layers:
        warnings.append("No layers added.")
        return warnings, 0, 0, 0, 0, 0, None
    
    # Collect bottom and top layers
    bottom_layers = []
    top_layers = []
    As_bottom = 0
    sum_As_y_bottom = 0
    As_top = 0
    sum_As_y_top = 0
    
    for layer in layers:
        bar_size = layer['bar_size']
        num_bars = layer['num_bars']
        dist = layer['dist']
        d_bar = rebar_diameters[bar_size]
        As_group = num_bars * bar_area(bar_size)
        
        # Cover check
        min_center_dist = (bottom_cover if layer['side'] == 'Bottom' else top_cover) + d_bar / 2
        if dist < min_center_dist:
            side_str = 'bottom' if layer['side'] == 'Bottom' else 'top'
            warnings.append(f"Insufficient {side_str} cover for layer at {dist:.2f} in: required min center dist {min_center_dist:.2f} in (ACI 20.5)")
        
        # For ratios
        if layer['side'] == 'Bottom':
            y = dist
            bottom_layers.append((y, d_bar))
            As_bottom += As_group
            sum_As_y_bottom += As_group * y
        else:
            y = dist  # dist from top
            top_layers.append((y, d_bar))
            As_top += As_group
            sum_As_y_top += As_group * y
        
        # Horizontal spacing check
        if num_bars > 1:
            center_spacing = (b - 2 * side_cover) / (num_bars - 1)
            clear_h = center_spacing - d_bar
            min_clear_h = max(1.0, d_bar)
            if clear_h < min_clear_h:
                warnings.append(f"Horizontal clear spacing {clear_h:.2f} in < required min {min_clear_h:.2f} in for {layer['side']} layer (ACI 25.2.1)")
    
    # Vertical spacing checks
    bottom_layers.sort(key=lambda x: x[0])
    if len(bottom_layers) > 1:
        for i in range(len(bottom_layers) - 1):
            y1, d1 = bottom_layers[i]
            y2, d2 = bottom_layers[i + 1]
            clear_v = y2 - y1 - (d1 / 2 + d2 / 2)
            min_clear_v = 1.0
            if clear_v < min_clear_v:
                warnings.append(f"Vertical clear spacing between bottom layers too small: {clear_v:.2f} in < {min_clear_v:.2f} in (ACI 25.2.2)")
    
    top_layers.sort(key=lambda x: x[0])
    if len(top_layers) > 1:
        for i in range(len(top_layers) - 1):
            y1, d1 = top_layers[i]
            y2, d2 = top_layers[i + 1]
            clear_v = y2 - y1 - (d1 / 2 + d2 / 2)
            min_clear_v = 1.0
            if clear_v < min_clear_v:
                warnings.append(f"Vertical clear spacing between top layers too small: {clear_v:.2f} in < {min_clear_v:.2f} in (ACI 25.2.2)")
    
    # Steel ratio checks
    if As_bottom == 0:
        warnings.append("No bottom reinforcement added.")
        rho = 0
        rho_prime = 0
        d = 0
        centroid_bottom = 0
    else:
        centroid_bottom = sum_As_y_bottom / As_bottom
        d = h - centroid_bottom
        rho = As_bottom / (b * d)
        rho_prime = As_top / (b * d) if As_top > 0 else 0
        
        # Convert fc and fy to psi
        fc_psi = fc * 1000
        fy_psi = fy * 1000
        rho_min = max(3 * math.sqrt(fc_psi) / fy_psi, 200 / fy_psi)
        if rho < rho_min:
            warnings.append(f"Bottom reinforcement ratio ρ = {rho:.4f} < minimum {rho_min:.4f} (ACI 9.6.1.1)")
        
        if rho > 0.04:
            warnings.append(f"Bottom reinforcement ratio ρ = {rho:.4f} > 0.04, may be impractical or over-reinforced.")
    
    if As_top > 0:
        centroid_top = sum_As_y_top / As_top
    else:
        centroid_top = None
    
    return warnings, rho, rho_prime, As_bottom, As_top, d, centroid_top

# Function to compute Phi Mn
def compute_phi_mn(fc, fy, b, d, As, As_prime, d_prime):
    beta1 = 0.85 if fc <= 4 else max(0.65, 0.85 - 0.05 * (fc - 4))
    Es = 29000.0
    ey = fy / Es
    epsilon_cu = 0.003
    
    if As_prime == 0:
        a = As * fy / (0.85 * fc * b)
        c = a / beta1
        fs_prime = 0
    else:
        # Assume compression steel yields
        a_assume = (As * fy - As_prime * fy) / (0.85 * fc * b)
        if a_assume < 0:
            return 0, 0, 0, "Too much compression steel."
        c_assume = a_assume / beta1
        epsilon_s_prime = epsilon_cu * (c_assume - d_prime) / c_assume if c_assume > d_prime else 0
        if epsilon_s_prime >= ey:
            fs_prime = fy
            a = a_assume
            c = c_assume
        else:
            alpha = 0.85 * fc * b * beta1
            gamma = As_prime * Es * epsilon_cu
            delta = gamma * d_prime
            A = alpha
            B = gamma - As * fy
            C = -delta
            discriminant = B**2 - 4 * A * C
            if discriminant < 0:
                return 0, 0, 0, "Invalid section."
            c = (-B + math.sqrt(discriminant)) / (2 * A)
            a = beta1 * c
            epsilon_s_prime = epsilon_cu * (c - d_prime) / c
            fs_prime = Es * epsilon_s_prime
    
    epsilon_t = epsilon_cu * (d - c) / c if c < d else 0
    
    extra_warning = ""
    if epsilon_t <= 0:
        phi = 0
        extra_warning = "Invalid strain distribution."
    elif epsilon_t >= 0.005:
        phi = 0.9
    elif epsilon_t >= 0.002:
        phi = 0.65 + (epsilon_t - 0.002) * (250 / 3)
        extra_warning = "Section is transition (not fully tension-controlled)."
    else:
        phi = 0.65
        extra_warning = "Section is compression-controlled."
    
    C_conc = 0.85 * fc * b * a
    C_steel = As_prime * fs_prime
    Mn_kip_in = C_conc * (d - a / 2) + C_steel * (d - d_prime)
    phi_mn_kip_in = phi * Mn_kip_in
    Mn = Mn_kip_in / 12.0  # report Mn in kip-ft
    phi_mn = phi_mn_kip_in / 12.0  # report φMn in kip-ft

    return phi, Mn, phi_mn, extra_warning

# Function to visualize the cross-section
def visualize_beam(b, h, layers, side_cover=1.5):
    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw beam rectangle
    ax.add_patch(plt.Rectangle((0, 0), b, h, fill=None, edgecolor='black', linewidth=1.5))
    
    # Draw rebars
    for layer in layers:
        side = layer['side']
        bar_size = layer['bar_size']
        num_bars = layer['num_bars']
        dist = layer['dist']
        d = rebar_diameters[bar_size]
        r = d / 2
        
        if side == 'Bottom':
            y = dist
        else:
            y = h - dist
        
        # Horizontal positions: evenly spaced with side covers
        if num_bars == 1:
            x_positions = [b / 2]
        else:
            center_spacing = (b - 2 * side_cover) / (num_bars - 1) if num_bars > 1 else 0
            x_positions = [side_cover + i * center_spacing for i in range(num_bars)]
        
        for x in x_positions:
            ax.add_patch(plt.Circle((x, y), r, fill=True, color='lightgray', edgecolor='black', linewidth=0.5))
    
    # Dimensions with CAD-like style
    # Width dimension below
    ax.plot([0, b], [-1, -1], color='black', linewidth=0.5)  # dim line
    ax.plot([0, 0], [-1.2, -0.8], color='black', linewidth=0.5)  # tick left
    ax.plot([b, b], [-1.2, -0.8], color='black', linewidth=0.5)  # tick right
    ax.text(b/2, -1.5, f"{b:.2f} in", ha='center', va='top', fontsize=8)
    
    # Height dimension right
    ax.plot([b+1, b+1], [0, h], color='black', linewidth=0.5)  # dim line
    ax.plot([b+0.8, b+1.2], [0, 0], color='black', linewidth=0.5)  # tick bottom
    ax.plot([b+0.8, b+1.2], [h, h], color='black', linewidth=0.5)  # tick top
    ax.text(b+1.5, h/2, f"{h:.2f} in", ha='left', va='center', fontsize=8, rotation=90)
    
    # Layer labels with leaders
    for layer in layers:
        side = layer['side']
        dist = layer['dist']
        if side == 'Bottom':
            y = dist
        else:
            y = h - dist
        
        # Draw leader line
        ax.plot([b, b+2], [y, y], color='black', linewidth=0.5)
        # Draw tick mark at beam edge
        ax.plot([b-0.2, b+0.2], [y-0.2, y+0.2], color='black', linewidth=0.5)
        ax.plot([b-0.2, b+0.2], [y+0.2, y-0.2], color='black', linewidth=0.5)
        # Add text label
        ax.text(b+2.2, y, f"{dist:.2f} in", ha='left', va='center', fontsize=8)
    
    ax.set_xlim(-2, b+4)
    ax.set_ylim(-2, h+2)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig

# Streamlit app
st.title("DESIGN LAYOUT - per ACI")

# Main layout with two columns: left for inputs, right for visualization
col_left, col_right = st.columns(2)

with col_left:
    col1, col2 = st.columns(2)
    with col1:
        b = st.number_input("b (in)", min_value=1.0, value=12.0, step=0.5, key='b_input')
        fc = st.number_input("f'c (ksi)", min_value=1.0, value=5.0, step=0.5, key='fc_input')
    with col2:
        h = st.number_input("h (in)", min_value=1.0, value=20.0, step=0.5, key='h_input')
        fy = st.number_input("fy (ksi)", min_value=1.0, value=60.0, step=5.0, key='fy_input')

    col_cover1, col_cover2, col_cover3 = st.columns(3)
    with col_cover1:
        bottom_cover = st.number_input("bot clr", min_value=0.0, value=1.5, step=0.25, key='bottom_cover')
    with col_cover2:
        top_cover = st.number_input("top clr", min_value=0.0, value=1.5, step=0.25, key='top_cover')
    with col_cover3:
        side_cover = st.number_input("side clr", min_value=0.0, value=1.5, step=0.25, key='side_cover')

    st.subheader("Bot Reinf")
    col_bottom1, col_bottom2 = st.columns(2)
    with col_bottom1:
        bar_size_bottom = st.selectbox("Bar size", list(rebar_diameters.keys()), key='bar_size_bottom')
        num_bars_bottom = st.number_input("# bars", min_value=0, value=2, step=1, key='num_bars_bottom')
    with col_bottom2:
        multi_layers_bottom = st.checkbox("Multiple layers", key='multi_bottom')
        if multi_layers_bottom:
            num_layers_bottom = st.number_input("# of layers", min_value=2, value=2, step=1, key='layers_bottom')
            spacing_bottom = st.number_input("C/C spacing (in)", min_value=0.0, value=2.0, step=0.25, key='spacing_bottom')
        else:
            num_layers_bottom = 1
            spacing_bottom = 0.0
        start_dist_bottom = st.number_input("Bot to center bar", min_value=0.0, value=2.5, step=0.25, key='start_bottom')

    st.subheader("Top Reinf")
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        bar_size_top = st.selectbox("Bar size", list(rebar_diameters.keys()), key='bar_size_top')
        num_bars_top = st.number_input("# bars", min_value=0, value=2, step=1, key='num_bars_top')
    with col_top2:
        multi_layers_top = st.checkbox("Multiple layers", key='multi_top')
        if multi_layers_top:
            num_layers_top = st.number_input("# of layers", min_value=2, value=2, step=1, key='layers_top')
            spacing_top = st.number_input("C/C spacing (in)", min_value=0.0, value=2.0, step=0.25, key='spacing_top')
        else:
            num_layers_top = 1
            spacing_top = 0.0
        start_dist_top = st.number_input("Top to center bar", min_value=0.0, value=2.5, step=0.25, key='start_top')

# Generate layers
layers = []

# Bottom layers
if num_bars_bottom > 0:
    for i in range(num_layers_bottom):
        dist = start_dist_bottom + i * spacing_bottom
        layers.append({
            'side': 'Bottom',
            'bar_size': bar_size_bottom,
            'num_bars': num_bars_bottom,
            'dist': dist
        })

# Top layers
if num_bars_top > 0:
    for i in range(num_layers_top):
        dist = start_dist_top + i * spacing_top
        layers.append({
            'side': 'Top',
            'bar_size': bar_size_top,
            'num_bars': num_bars_top,
            'dist': dist
        })

# Check validity and display warnings, ratios
warnings, rho_bottom, rho_top, As_bottom, As_top, d, d_prime = check_validity(b, h, fc, fy, layers, side_cover, bottom_cover, top_cover)

# Visualize cross-section in the right column if layers exist
with col_right:
    if layers:
        fig = visualize_beam(b, h, layers, side_cover)
        st.pyplot(fig)
    
    st.write(f"ρ_bot: {rho_bottom:.4f}")
    st.write(f"ρ_top: {rho_top:.4f}")
    
    if warnings:
        st.subheader("Warnings")
        for w in warnings:
            st.warning(w)
    
    if st.button("Calculate Phi*Mn"):
        if As_bottom > 0:
            As_prime = As_top if As_top > 0 else 0
            d_prime_val = d_prime if d_prime is not None else 0
            phi, Mn, phi_mn, extra_warning = compute_phi_mn(fc, fy, b, d, As_bottom, As_prime, d_prime_val)
            st.write(f"Phi = {phi:.2f}")
            st.write(f"Mn = {Mn:.2f} kip-ft")
            st.write(f"Phi Mn = {phi_mn:.2f} kip-ft")
            if extra_warning:
                st.warning(extra_warning)
        else:
            st.write("No bottom reinforcement.")
