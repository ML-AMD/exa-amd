import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from pymatgen.core import Composition, Element
from scipy.spatial import ConvexHull

# --- Tetrahedral Projection ---
# Define standard coordinates for the 4 vertices (representing pure elements)
# A at (0, 0, 0), B at (1, 0, 0), C at (0.5, sqrt(3)/2, 0), D at (0.5,
# sqrt(3)/6, sqrt(6)/3)
TETRA_CORNERS = {
    0: np.array([0.0, 0.0, 0.0]),                   # Element A (index 0)
    1: np.array([1.0, 0.0, 0.0]),                   # Element B (index 1)
    2: np.array([0.5, np.sqrt(3) / 2, 0.0]),          # Element C (index 2)
    3: np.array([0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]),  # Element D (index 3)
}


def composition_to_tetrahedral_coords(comp, element_map):
    """
    Converts a pymatgen Composition object to 3D tetrahedral coordinates.

    Args:
        comp (Composition): The composition to convert.
        element_map (dict): A dictionary mapping Element objects to their
                            vertex index (0, 1, 2, 3).

    Returns:
        np.ndarray: The 3D coordinates (x, y, z), or None if composition
                    doesn't match the 4 elements.
    """
    coords = np.zeros(3)
    total_fraction = 0.0
    try:
        # Calculate weighted average of corner coordinates based on atomic fractions
        # We skip the element at index 0 as it's implicitly represented
        # (origin)
        for element, index in element_map.items():
            fraction = comp.get_atomic_fraction(element)
            if index != 0:  # Don't add contribution from the origin element explicitly
                coords += fraction * TETRA_CORNERS[index]
            total_fraction += fraction

        # Basic check if the composition belongs to the system
        if not np.isclose(total_fraction, 1.0):
            # This might happen if comp contains elements outside the map
            # Or if it's an empty composition
            # print(f"Warning: Composition {comp.reduced_formula} fractions don't sum to 1 for the system. Skipping.")
            return None
        # Check if all elements in the comp are in our system
        if not all(el in element_map for el in comp.elements):
            return None

        return coords
    except Exception as e:
        print(f"Error converting composition {comp.reduced_formula}: {e}")
        return None

# --- Data Parsing Functions ---


def parse_stable_phases(filename, element_map):
    """
    Parses the stable phases file (e.g., mp_int_stable.dat).

    Args:
        filename (str): Path to the stable phases file.
        element_map (dict): Dictionary mapping Element objects to vertex indices.

    Returns:
        list: A list of tuples, where each tuple is
              (formula, energy_per_atom, np.ndarray_coords).
              Returns only phases belonging to the A-B-C-D system.
    """
    stable_phases = []
    elements_in_system = set(element_map.keys())
    print(f"Parsing stable phases from: {filename}")
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                formula = parts[0]
                try:
                    energy = float(parts[-1])  # Assume energy is the last part
                    comp = Composition(formula)

                    # Check if the composition's elements are a subset of our
                    # system
                    if set(comp.elements).issubset(elements_in_system):
                        coords = composition_to_tetrahedral_coords(
                            comp, element_map)
                        if coords is not None:
                            stable_phases.append((formula, energy, coords))
                        # else:
                            # print(f"  Skipping stable phase {formula} (coord conversion failed or outside system).")
                    # else:
                        # print(f"  Skipping stable phase {formula} (elements outside system).")

                except (ValueError, TypeError) as e:
                    print(
                        f"  Warning: Could not parse line: '{line}'. Error: {e}")
                except Exception as e:
                    print(
                        f"  Warning: Could not process composition {formula}: {e}")

    except FileNotFoundError:
        print(f"Error: Stable phases file '{filename}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred reading {filename}: {e}")
        return []

    print(
        f"Found {
            len(stable_phases)} stable phases within the specified element system.")
    return stable_phases


def parse_results_csv(filename, element_map):
    """
    Parses the results CSV file (e.g., *_quaternary.csv).
    Assumes columns: Formula,Total_Energy_per_atom,Ehull,...

    Args:
        filename (str): Path to the results CSV file.
        element_map (dict): Dictionary mapping Element objects to vertex indices.

    Returns:
        list: A list of tuples, where each tuple is
              (formula, ehull, np.ndarray_coords).
    """
    results = []
    print(f"Parsing calculated results from: {filename}")
    try:
        with open(filename, 'r') as f:
            header = f.readline().strip().lower()  # Read header
            if not header.startswith('formula'):
                print(
                    "Warning: CSV file does not seem to have the expected header (Formula,...). Trying to parse anyway.")

            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 3:  # Need at least Formula, Total_Energy, Ehull
                    print(f"  Warning: Skipping malformed line: '{line}'")
                    continue
                formula = parts[0]
                try:
                    # Ehull is expected to be the 3rd column (index 2)
                    ehull = float(parts[2])
                    comp = Composition(formula)
                    coords = composition_to_tetrahedral_coords(
                        comp, element_map)
                    if coords is not None:
                        results.append((formula, ehull, coords))

                except (ValueError, TypeError) as e:
                    print(
                        f"  Warning: Could not parse Ehull or composition for line: '{line}'. Error: {e}")
                except Exception as e:
                    print(
                        f"  Warning: Could not process composition {formula} from results: {e}")

    except FileNotFoundError:
        print(f"Error: Results file '{filename}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred reading {filename}: {e}")
        return []

    print(
        f"Found {
            len(results)} calculated results within the specified element system.")
    return results

# --- Plotting Function ---


def plot_quaternary_hull(elements_str, stable_phases,
                         calculated_results, ehull_threshold, output_file=None):
    """
    Generates the 3D plot of the quaternary convex hull.

    Args:
        elements_str (list): List of 4 element symbols (e.g., ['Si','Ge','Sn','Pb']).
        stable_phases (list): List of (formula, energy, coords) for stable phases.
        calculated_results (list): List of (formula, ehull, coords) for calculated phases.
        ehull_threshold (float): Max Ehull value to plot for calculated phases.
        output_file (str, optional): Path to save the plot image. If None, displays plot.
    """
    if len(elements_str) != 4:
        raise ValueError("Exactly 4 element symbols are required.")

    elements = [Element(el) for el in elements_str]
    # Create the mapping from Element object to vertex index (0, 1, 2, 3)
    element_map = {el: i for i, el in enumerate(elements)}

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print("Plotting tetrahedron edges...")
    corners_3d = list(TETRA_CORNERS.values())
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        ax.plot([corners_3d[i][0], corners_3d[j][0]],
                [corners_3d[i][1], corners_3d[j][1]],
                [corners_3d[i][2], corners_3d[j][2]], 'k-', lw=1.0, alpha=0.6)

    # Label corners
    corner_labels = elements_str
    for i, label in enumerate(corner_labels):
        ax.text(corners_3d[i][0] * 1.05, corners_3d[i][1] * 1.05, corners_3d[i][2] * 1.05, label,
                fontsize=15, ha='center', va='center')

    print("Computing and plotting convex hull facets...")
    stable_coords = np.array([p[2] for p in stable_phases if p[2] is not None])

    if len(stable_coords) >= 4:  # Need at least 4 points for a 3D hull
        try:
            hull = ConvexHull(stable_coords)
            # Plot the triangular faces of the hull
            for simplex in hull.simplices:
                triangle = stable_coords[simplex]
                face = Poly3DCollection(
                    [triangle],
                    alpha=0.2,
                    facecolor='lightblue',
                    edgecolor='grey',
                    lw=0.5)
                ax.add_collection3d(face)
            print(
                f"  Successfully computed and plotted hull with {len(hull.simplices)} facets.")
        except Exception as e:
            print(
                f"  Warning: Could not compute or plot convex hull: {e}. Only plotting points.")
            # Proceed without plotting hull faces if computation fails
    else:
        print("  Warning: Not enough stable points (need >= 4) to compute 3D convex hull.")

    print("Plotting stable phase points...")
    if stable_coords.any():  # Check if there are any stable coordinates to plot
        ax.scatter(stable_coords[:, 0], stable_coords[:, 1], stable_coords[:, 2],
                   c='black', marker='o', s=60, label='Stable Phases (Input)', depthshade=False, alpha=0.8)
    else:
        print("  No stable phase coordinates found to plot.")

    print(
        f"Plotting calculated results with Ehull <= {ehull_threshold} eV/atom...")
    calculated_coords = []
    calculated_ehull = []
    calculated_labels = []

    for formula, ehull, coords in calculated_results:
        if coords is not None and ehull <= ehull_threshold:
            calculated_coords.append(coords)
            calculated_ehull.append(ehull)
            # Keep label for potential hover/annotation later
            calculated_labels.append(formula)

    if calculated_coords:
        calculated_coords = np.array(calculated_coords)
        calculated_ehull = np.array(calculated_ehull)

        # Normalize Ehull values for colormap
        norm = mcolors.Normalize(vmin=0, vmax=ehull_threshold)
        # Reversed viridis: blue (low Ehull) to yellow (high Ehull)
        cmap = plt.cm.rainbow_r

        sc = ax.scatter(calculated_coords[:, 0], calculated_coords[:, 1], calculated_coords[:, 2],
                        c=calculated_ehull, cmap=cmap, norm=norm,
                        marker='^', s=40, label=f'Calculated (Ehull <= {ehull_threshold:.3f})',
                        depthshade=True, alpha=0.9)  # Use depthshade for better 3D perception

        # Add Colorbar
        cbar = fig.colorbar(sc, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('Formation Energy above Hull (eV/atom)')
        print(f"  Plotted {len(calculated_coords)} calculated points.")
    else:
        print("  No calculated results found within the Ehull threshold.")

    ax.set_xlabel('Composition Space X')
    ax.set_ylabel('Composition Space Y')
    ax.set_zlabel('Composition Space Z')

    # Remove axis ticks/grid for cleaner compositional space view
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.axis('off')  # Turn off the axis frame

    # Adjust view angle (elevation, azimuth)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize a quaternary convex hull.')
    parser.add_argument(
        "--elements",
        type=str,
        required=True,
        help="Elements (4 required, separated by hyphen, e.g., A-B-C-D)")
    parser.add_argument('--stable', required=True, type=str,
                        help='Path to the stable phases file (e.g., mp_int_stable.dat)')
    parser.add_argument('--input', required=True, type=str,
                        help='Path to the calculated results CSV file (e.g., SiGeSnPb_quaternary.csv)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Maximum Ehull (eV/atom) to display for metastable phases (default: 0.1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file name (e.g., hull_plot.png). If not provided, shows plot interactively.')

    args = parser.parse_args()
    current_dir = os.path.basename(os.getcwd())

    element_symbols = args.elements.split('-')
    if len(element_symbols) != 4:
        parser.error("Exactly four elements must be provided via --elements.")

    # Create element map {Element('Si'): 0, Element('Ge'): 1, ...}
    element_map_main = {Element(el): i for i, el in enumerate(element_symbols)}

    # Use provided elements
    if args.input is None:
        args.input = os.path.basename(os.getcwd()) + '.csv'

    results_data = parse_results_csv(args.input, element_map_main)

    if args.output is None:
        args.output = os.path.basename(os.getcwd()) + '.png'

    # Parse the data
    stable_data = parse_stable_phases(args.stable, element_map_main)

    # Generate the plot
    if not stable_data and not results_data:
        print("\nError: No data could be parsed from input files for the specified elements. Cannot generate plot.")
    else:
        plot_quaternary_hull(
            element_symbols,
            stable_data,
            results_data,
            args.threshold,
            args.output)
