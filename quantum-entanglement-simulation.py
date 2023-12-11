import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.special import expit

# Function to calculate gravitational field
def gravitational_field(position):
    G = 6.67430e-11
    mass_of_earth = 5.972e24
    magnitude = np.linalg.norm(position)
    return G * (mass_of_earth / magnitude**2)

# Function for quantum gravitational field (using placeholders, replace with real model)
def quantum_gravitational_field(position, quantum_params):
    return gravitational_field(position) * quantum_params[0]

# Function to calculate entanglement probability
def calculate_entanglement_probability(gravitational_field_strength, params):
    k, x0 = params
    return expit(k * (gravitational_field_strength - x0))

# Function to initialize animation
def init():
    scatter.set_sizes([])  # Initialize empty sizes
    scatter.set_offsets(np.array([]).reshape(0, 3))  # Initialize empty positions
    line.set_data([], [])  # Initialize line data
    return scatter, line

def update_entanglement_probability(num, data, scatter, quantum_params, params, line):
    # Update positions dynamically
    dynamic_positions = data[:num+1, :]

    # Calculate quantum gravitational field at each dynamic position
    quantum_field_values = np.array([quantum_gravitational_field(pos, quantum_params) for pos in dynamic_positions])

    # Calculate entanglement probabilities for dynamic quantum field values
    entanglement_probabilities = np.array([calculate_entanglement_probability(grav_field, params)
                                           for grav_field in quantum_field_values])

    sizes = entanglement_probabilities * 100
    sizes = np.clip(sizes, 10, 100)  # Adjust size range as needed

    scatter.set_offsets(np.column_stack([dynamic_positions[:, 0], dynamic_positions[:, 1], dynamic_positions[:, 2]]))
    scatter.set_sizes(sizes)

    ax.set_title(f'Frame {num+1}/{len(data)}')

    # Visualization of the relationship between quantum gravitational field and entanglement probability
    line.set_xdata(normalized_gravitational_values[:num+1])
    line.set_ydata(entanglement_probabilities)

    # Clear previous sizes and reset sizes with the new data
    scatter._sizes = []
    scatter._sizes3d = np.array([]).reshape(0, 1)
    for s in sizes:
        scatter._sizes.append(s)
        scatter._sizes3d = np.vstack([scatter._sizes3d, np.ones_like(scatter._sizes3d[:1]) * s])

    ax2.relim()
    ax2.autoscale_view()

    return scatter, line

# Generate random positions within Earth's atmosphere
np.random.seed(0)
num_points = 500
positions = np.random.rand(num_points, 3) * 6371000  # Asegúrate de que positions sea bidimensional

# Calculate gravitational field at each position
gravitational_values = np.array([gravitational_field(pos) for pos in positions])

# Normalize gravitational field values
normalized_gravitational_values = (gravitational_values - gravitational_values.min()) / (
    gravitational_values.max() - gravitational_values.min())

# Random quantum parameters
quantum_params = np.random.rand(1)

# Sigmoid function parameters
sigmoid_params = [1, 0]  # Adjust as needed

# Set up figure and 3D axes
figure = plt.figure(figsize=(18, 8))
ax = figure.add_subplot(121, projection='3d')

# 3D scatter plot with colors based on gravitational field
scatter = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    positions[:, 2],
    c=positions[:, 2],
    cmap='viridis',
    marker='o',
    alpha=0.8,
    edgecolors='k',
    linewidths=0.5,
)

# Color bar
color_bar = figure.colorbar(scatter, ax=ax, pad=0.1)
color_bar.set_label('Z-axis Position')

# Axis labels and title
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')

# Set axis limits to maintain constant scale
ax.set_xlim([0, 6371000])
ax.set_ylim([0, 6371000])
ax.set_zlim([0, 6371000])

# Improve plot with grid, better layout, and enhanced perspective
ax.grid(True)

# Second subplot for entanglement probability visualization
ax2 = figure.add_subplot(122)
line, = ax2.plot([], [], lw=2)  # Cambiado a un gráfico de líneas
ax2.set_xlabel('Normalized Gravitational Field')
ax2.set_ylabel('Entanglement Probability')
ax2.set_title('Quantum Gravitational Field vs. Entanglement Probability')
ax2.grid(True)

# Add legends and additional labels
ax.legend(['Earth Positions'], loc='upper right')
ax2.legend(['Entanglement Probability'], loc='upper right')

# Create animation
ani = animation.FuncAnimation(
    figure, update_entanglement_probability, frames=len(normalized_gravitational_values),
    fargs=(positions, scatter, quantum_params, sigmoid_params, line),
    init_func=init,
    interval=50, blit=False
)

plt.ion()
plt.show(block=True)