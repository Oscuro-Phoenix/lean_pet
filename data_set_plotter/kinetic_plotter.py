import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.lines as mlines
import warnings
import os

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set plot style to match the provided image
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

fig, ax = plt.subplots(figsize=(8, 6))

datasets = [
    {'file': 'LCO_Ando_2023.csv', 'material': 'LCO', 'label': 'Ando et al. (2023)', 'scale': 1e4},
    {'file': 'LCO_Xiong_2023.csv', 'material': 'LCO', 'label': 'Xiong et al. (2023)', 'scale': 1.0},
    {'file': 'LCO_Zhang_2025.csv', 'material': 'LCO', 'label': 'Zhang et al. (2025)', 'scale': 1e3},
    {'file': 'NMC111_Zhang_2025.csv', 'material': 'NMC111', 'label': 'Zhang et al. (2025)', 'scale': 1e3},
    {'file': 'NMC532_Xiong_2023.csv', 'material': 'NMC532', 'label': 'Xiong et al. (2023)', 'scale': 1.0},
    {'file': 'NCA_Ando_2018.csv', 'material': 'NCA', 'label': 'Ando et al. (2018)', 'scale': 1e4},
]

# Create a color map based on unique materials
unique_materials = ['LCO', 'NMC111', 'NMC532', 'NCA']
color_map = {material: color for material, color in zip(unique_materials, cm.coolwarm(np.linspace(0.1, 0.9, len(unique_materials))))}

# Create a marker map for sources
unique_sources = ['Ando et al. (2018)', 'Ando et al. (2023)', 'Xiong et al. (2023)', 'Zhang et al. (2025)']
markers_list = ['*', 'o', 's', '^'] # map distinct markers to unique sources
marker_map = {source: marker for source, marker in zip(unique_sources, markers_list)}

for d in datasets:
    file = d['file']
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue
    
    # Read the data, assuming no header
    df = pd.read_csv(file, header=None)
    print(f"Processing {file}...")
    
    # User specified: x is overpotential, y is current
    x = df[0]
    y = df[1]

    # Scale each dataset by its maximum magnitude current value
    max_magnitude = np.max(np.abs(y))
    y_transformed = (y / max_magnitude) * np.sign(x)

    # Filter data for linear fit (kinetics typically within ±0.17 V overpotential)
    fit_mask = (x >= -0.17) & (x <= 0.17)
    
    # Perform linear fit on the filtered data
    if fit_mask.sum() > 1: # Need at least 2 points to fit a line
        slope, intercept = np.polyfit(x[fit_mask], y_transformed[fit_mask], 1)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color=color_map[d['material']], linewidth=2)

    # Plot data as scatter points
    ax.scatter(x, y_transformed,
               marker=marker_map[d['label']],
               s=100,  # size of marker
               facecolors=color_map[d['material']],
               edgecolors='black',
               linewidth=1)

# Set axis labels to match the image
ax.set_xlabel('Overpotential (V)', fontsize=22)
ax.set_ylabel('Current Density (scaled)', fontsize=22)

# Create colored material list text
x_pos, y_pos = 0.05, 0.95
ax.text(x_pos, y_pos, "Materials:", transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left', va='top')
x_pos += 0.22  # Starting position for the first material

for i, material in enumerate(unique_materials):
    color = color_map[material]
    label = f'{material}'
    if i < len(unique_materials) - 1:
        label += ','

    t = ax.text(x_pos, y_pos, label, color=color, transform=ax.transAxes, fontsize=16, fontweight='bold', ha='left', va='top')
    x_pos += (len(material) * 0.025) + 0.04 # Adjust spacing for next label

# Add padding to the plot
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_padding = (xlim[1] - xlim[0]) * 0.1
y_padding = (ylim[1] - ylim[0]) * 0.1
ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

# Create legend for sources
legend_handles = []
for source, marker in marker_map.items():
    legend_handles.append(mlines.Line2D([], [], color='gray', marker=marker, linestyle='None',
                                        markersize=10, label=source))

ax.legend(handles=legend_handles, loc='lower right', fontsize=15)
plt.tight_layout()
plt.savefig('kinetic_plot_v2.png', dpi=300)
plt.show()
