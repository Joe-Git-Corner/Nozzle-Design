import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from bisect import bisect_left
import csv

"""
Docstring from original file retained for reference:
 Implemented from the following technical notes 
 The thrust optimised parabolic nozzle
 http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf
 ... [Equations omitted for brevity] ...
"""

# =============================================================
# Helper Functions (Interpolation, Nearest Value)
# =============================================================

def interpolate(x_list, y_list, x):
    """Simple linear interpolation."""
    if not all(y > x_val for x_val, y in zip(x_list, x_list[1:])):
         # Check if strictly ascending, allow duplicates for edge cases
         if len(set(x_list)) != len(x_list):
             pass # Allow duplicate x values if needed, might happen at edges
         elif not all(y >= x_val for x_val, y in zip(x_list, x_list[1:])):
             raise ValueError("x_list must be in ascending order!")
             
    # Handle extrapolation cases
    if x <= x_list[0]:
        return y_list[0]
    if x >= x_list[-1]:
        return y_list[-1]

    # Find interval
    i = bisect_left(x_list, x) - 1
    
    # Calculate slope - handle potential division by zero if x values are duplicated
    x1, x2 = x_list[i], x_list[i+1]
    y1, y2 = y_list[i], y_list[i+1]
    
    if x2 == x1:
        # If x values are identical, return the average or first y value
        return (y1 + y2) / 2
    else:
        slope = (y2 - y1) / (x2 - x1)
        return y_list[i] + slope * (x - x_list[i])


def find_nearest(array, value):
    """Find the nearest value and index in the list/array for the given value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# =============================================================
# Core Bell Nozzle Calculation Functions
# =============================================================

def find_wall_angles(ar, Rt_mm, l_percent=80):
    """
    Find wall angles (theta_n, theta_e) in radians and nozzle length (Ln) in mm
    for a given area ratio (ar), throat radius (Rt_mm), and length percentage.
    Uses empirical data and interpolation based on Rao's method.
    """
    # Wall-angle empirical data (degrees)
    aratio_data = [ 4,    5,    10,   20,   30,   40,   50,   100]
    theta_n_60  = [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
    theta_n_80  = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
    theta_n_90  = [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
    theta_e_60  = [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
    theta_e_80  = [14.0, 13.0, 11.0,  9.0,  8.5,  8.0,  7.5,  7.0]
    theta_e_90  = [11.5, 10.5,  8.0,  7.0,  6.5,  6.0,  6.0,  6.0]

    # Reference angle for length calculation (15 degrees)
    ref_angle_rad = math.radians(15)

    # Calculate base length factor L' = (sqrt(epsilon) - 1) * Rt / tan(15deg)
    try:
        base_length_factor = (math.sqrt(ar) - 1) * Rt_mm / math.tan(ref_angle_rad)
    except ValueError:
        print(f"Warning: Invalid input for sqrt in find_wall_angles (ar={ar}). Returning default angles.")
        return 0, math.radians(30), math.radians(8) # Return some default values

    # Select angle data and calculate nozzle length (Ln) based on l_percent
    if l_percent == 60:
        theta_n_data = theta_n_60
        theta_e_data = theta_e_60
        Ln = 0.6 * base_length_factor
    elif l_percent == 80:
        theta_n_data = theta_n_80
        theta_e_data = theta_e_80
        Ln = 0.8 * base_length_factor
    elif l_percent == 90:
        theta_n_data = theta_n_90
        theta_e_data = theta_e_90
        Ln = 0.9 * base_length_factor
    else: # Default to 80% if input is invalid
        print(f"Warning: Invalid l_percent ({l_percent}). Defaulting to 80%.")
        theta_n_data = theta_n_80
        theta_e_data = theta_e_80
        Ln = 0.8 * base_length_factor
        l_percent = 80 # Ensure consistency

    # Interpolate to find angles for the specific area ratio 'ar'
    try:
        tn_deg = interpolate(aratio_data, theta_n_data, ar)
        te_deg = interpolate(aratio_data, theta_e_data, ar)
    except ValueError as e:
         print(f"Error during angle interpolation: {e}. Returning default angles.")
         return Ln, math.radians(30), math.radians(8) # Return default values on error

    # Convert angles to radians
    theta_n_rad = math.radians(tn_deg)
    theta_e_rad = math.radians(te_deg)

    return Ln, theta_n_rad, theta_e_rad


def bell_nozzle(k, aratio, Rt, l_percent):
    """
    Calculates the contour coordinates for a Rao bell nozzle.

    Args:
        k (float): Ratio of specific heats (gamma). Not directly used in Rao contour calc,
                   but often passed along. Included for consistency if needed later.
        aratio (float): Nozzle area ratio (Ae/At).
        Rt (float): Throat radius (ensure units are consistent, e.g., mm).
        l_percent (int): Nozzle length percentage (e.g., 60, 80, 90).

    Returns:
        tuple: (angles, contour_data)
            angles (tuple): (nozzle_length, theta_n_rad, theta_e_rad) in consistent units (e.g., mm, rad).
            contour_data (tuple): Lists of coordinates for plotting:
                                  (xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell)
    """
    # Entrant angle (typically -135 degrees)
    entrant_angle_deg = -135
    ea_radian = math.radians(entrant_angle_deg)

    # Find wall angles and nozzle length based on inputs
    # Rt is expected in mm by find_wall_angles based on original context
    nozzle_length, theta_n_rad, theta_e_rad = find_wall_angles(aratio, Rt, l_percent)
    angles = (nozzle_length, theta_n_rad, theta_e_rad)

    data_interval = 100 # Number of points per section

    # --- Entrant section (Circular Arc 1) ---
    # Based on Eqn. 4: x = 1.5*Rt*cos(theta), y = 1.5*Rt*sin(theta) + 1.5*Rt + Rt
    # Center: (0, 1.5*Rt + Rt) = (0, 2.5*Rt) ? Typo in paper, should be (0, 1.5Rt)? Assuming (0, 1.5Rt+Rt) = (0, 2.5Rt) per Eq4 text
    # Radius: 1.5*Rt
    # Angle range: entrant_angle_deg to -90 deg
    ea_start = ea_radian
    ea_end = -math.pi / 2
    angle_list_e1 = np.linspace(ea_start, ea_end, data_interval)
    xe = 1.5 * Rt * np.cos(angle_list_e1)
    # Using + 2.5 * Rt based on formula text, although center might be intended differently.
    # This defines the Y coordinate relative to the nozzle centerline.
    # If centerline is y=0, center is (0, 2.5*Rt), y = y_center + R*sin(theta)
    # Let's re-read the paper: "y = 1.5 Rt sinθ + 1.5 Rt + Rt" implies y relative to some origin.
    # If origin is chamber end, centerline y=0, then maybe origin x=0 is throat?
    # Sticking to formula as written: y = 1.5*Rt*sin(theta) + 2.5*Rt
    # Check plot consistency: If throat y=Rt, then 2.5Rt seems high for center.
    # Let's assume center is (0, 1.5*Rt), so y = 1.5*Rt*sin(theta) + 1.5*Rt
    # UPDATE: Original script used `ye.append( 1.5 * Rt * math.sin(i) + 2.5 * Rt )`. Sticking to that.
    ye = 1.5 * Rt * np.sin(angle_list_e1) + 2.5 * Rt
    # Correction: Plot in original seems to have y=Rt at throat (x=0).
    # Eqn 4: y = 1.5Rt sin(theta) + 1.5Rt + Rt. If y(theta=-90)=Rt, then 1.5Rt*(-1) + 1.5Rt + Rt = Rt. This matches.
    
    # --- Throat exit section (Circular Arc 2) ---
    # Based on Eqn. 5: x = 0.382*Rt*cos(theta), y = 0.382*Rt*sin(theta) + 0.382*Rt + Rt
    # Center: (0, 0.382*Rt + Rt) = (0, 1.382*Rt)
    # Radius: 0.382*Rt
    # Angle range: -90 deg to (theta_n_deg - 90) deg
    ea_start_e2 = -math.pi / 2
    ea_end_e2 = theta_n_rad - math.pi / 2 # theta_n is already in radians
    angle_list_e2 = np.linspace(ea_start_e2, ea_end_e2, data_interval)
    xe2 = 0.382 * Rt * np.cos(angle_list_e2)
    ye2 = 0.382 * Rt * np.sin(angle_list_e2) + 1.382 * Rt
    # Check continuity: At theta=-90, x=0, y=0.382*(-1)+1.382 = 1.0*Rt. Matches Arc 1 if that ends at y=Rt.
    # Check point N coords: x=0.382*cos(theta_n-90), y=0.382*sin(theta_n-90)+1.382*Rt matches below.

    # --- Bell section (Quadratic Bezier Curve) ---
    # Point N (end of Arc 2)
    Nx = 0.382 * Rt * math.cos(theta_n_rad - math.pi / 2)
    Ny = 0.382 * Rt * math.sin(theta_n_rad - math.pi / 2) + 1.382 * Rt

    # Point E (nozzle exit)
    Ex = nozzle_length # Ln calculated by find_wall_angles
    Ey = math.sqrt(aratio) * Rt # From Eqn. 2

    # Point Q (intersection of tangent lines)
    # Gradients m1, m2 - Eqn. 8 (using radians)
    m1 = math.tan(theta_n_rad)
    m2 = math.tan(theta_e_rad)

    # Intercepts C1, C2 - Eqn. 9
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex

    # Intersection Q - Eqn. 10
    if abs(m1 - m2) < 1e-9: # Avoid division by zero if angles are identical
        print("Warning: Bell nozzle tangent angles theta_n and theta_e are too close. Check inputs.")
        # Approximate Q or handle error - using midpoint for now
        Qx = (Nx + Ex) / 2
        Qy = (Ny + Ey) / 2
    else:
        Qx = (C2 - C1) / (m1 - m2)
        Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Bezier curve calculation - Eqn. 6
    t_list = np.linspace(0, 1, data_interval)
    xbell = ((1 - t_list)**2) * Nx + 2 * (1 - t_list) * t_list * Qx + (t_list**2) * Ex
    ybell = ((1 - t_list)**2) * Ny + 2 * (1 - t_list) * t_list * Qy + (t_list**2) * Ey

    # Create negative y-values for the lower half of the nozzle
    nye = -ye
    nye2 = -ye2
    nybell = -ybell

    contour_data = (xe.tolist(), ye.tolist(), nye.tolist(),
                    xe2.tolist(), ye2.tolist(), nye2.tolist(),
                    xbell.tolist(), ybell.tolist(), nybell.tolist())

    return angles, contour_data


# =============================================================
# Plotting Functions
# =============================================================

def draw_angle_arc(ax, angle_rad, origin, label=r'$\theta$', radius=10, text_offset=0.5):
    """Draws an angle arc and label on the plot."""
    start_point = np.array(origin)
    angle_deg = math.degrees(angle_rad)

    # Create arc patch
    # Note: Arc angles are in degrees
    arc_patch = Arc(start_point, radius*2, radius*2, angle=0,
                    theta1=0, theta2=angle_deg, color='k', linewidth=0.5)
    ax.add_patch(arc_patch)

    # Add line for angle reference if needed (visual aid)
    # end_point = start_point + radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    # ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'k:', linewidth=0.5)
    # ax.plot([start_point[0], start_point[0]+radius], [start_point[1], start_point[1]], 'k:', linewidth=0.5) # Horizontal line

    # Add text label
    text_angle_rad = angle_rad / 2 # Place text in middle of arc
    text_pos = start_point + (radius + text_offset) * np.array([math.cos(text_angle_rad), math.sin(text_angle_rad)])
    ax.text(text_pos[0], text_pos[1], f'{label} = {angle_deg:.1f}°',
            horizontalalignment='center', verticalalignment='center', fontsize=8)


def plot_nozzle(ax, title, Rt, angles, contour):
    """Plots the 2D nozzle contour with annotations."""
    # Unpack angles and contour data
    nozzle_length, theta_n_rad, theta_e_rad = angles
    xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell = contour
    Re = ybell[-1] # Exit radius

    # Plot contours
    ax.plot(xe, ye, linewidth=1.5, color='blue', label='Entrant Arc')
    ax.plot(xe, nye, linewidth=1.5, color='blue')
    ax.plot(xe2, ye2, linewidth=1.5, color='red', label='Throat Exit Arc')
    ax.plot(xe2, nye2, linewidth=1.5, color='red')
    ax.plot(xbell, ybell, linewidth=1.5, color='green', label='Bell Curve')
    ax.plot(xbell, nybell, linewidth=1.5, color='green')

    # Set aspect ratio and axis lines
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', lw=0.5, linestyle='--') # Centerline
    ax.axvline(0, color='black', lw=0.5, linestyle='--') # Throat line

    # --- Annotations ---
    # Throat radius
    ax.annotate(f'Rt = {Rt:.2f}', xy=(0, Rt), xytext=(-Rt*1.5, Rt * 1.2),
                arrowprops=dict(arrowstyle='->', lw=0.5), fontsize=8)
    ax.plot([0, 0], [0, Rt], 'k:', lw=0.5)

    # Exit radius
    ax.annotate(f'Re = {Re:.2f}', xy=(nozzle_length, Re), xytext=(nozzle_length*0.8, Re * 1.1),
                arrowprops=dict(arrowstyle='->', lw=0.5), fontsize=8)
    ax.plot([nozzle_length, nozzle_length], [0, Re], 'k:', lw=0.5)

    # Nozzle length
    ax.annotate(f'Ln = {nozzle_length:.2f}', xy=(nozzle_length/2, 0), xytext=(nozzle_length/2, -Re*0.3),
                arrowprops=dict(arrowstyle='->', lw=0.5), fontsize=8)
    ax.plot([0, nozzle_length], [0, 0], 'k:', lw=0.5)

    # Theta_n (inflection angle) - draw at point N (end of xe2)
    N_point = (xe2[-1], ye2[-1])
    # Need tangent line for reference, slope is tan(theta_n_rad)
    # Draw a short tangent line segment
    tan_len = Rt * 0.5
    tan_x = [N_point[0] - tan_len*np.cos(theta_n_rad), N_point[0] + tan_len*np.cos(theta_n_rad)]
    tan_y = [N_point[1] - tan_len*np.sin(theta_n_rad), N_point[1] + tan_len*np.sin(theta_n_rad)]
    # ax.plot(tan_x, tan_y, 'm--', lw=0.5) # Draw tangent if needed
    # Draw angle arc relative to horizontal at point N
    draw_angle_arc(ax, theta_n_rad, N_point, label=r'$\theta_n$', radius=Rt*0.6, text_offset=Rt*0.1)


    # Theta_e (exit angle) - draw at point E (end of xbell)
    E_point = (xbell[-1], ybell[-1])
     # Draw a short tangent line segment
    tan_len_e = Rt * 0.5
    tan_xe = [E_point[0] - tan_len_e*np.cos(theta_e_rad), E_point[0] + tan_len_e*np.cos(theta_e_rad)]
    tan_ye = [E_point[1] - tan_len_e*np.sin(theta_e_rad), E_point[1] + tan_len_e*np.sin(theta_e_rad)]
    # ax.plot(tan_xe, tan_ye, 'c--', lw=0.5) # Draw tangent if needed
    # Draw angle arc relative to horizontal at point E
    draw_angle_arc(ax, theta_e_rad, E_point, label=r'$\theta_e$', radius=Rt*0.6, text_offset=Rt*0.1)

    # Arc Radii (optional - can clutter plot)
    # ax.annotate(f'R1 = {1.5*Rt:.1f}', xy=(xe[len(xe)//2], ye[len(ye)//2]), ...)
    # ax.annotate(f'R2 = {0.382*Rt:.1f}', xy=(xe2[len(xe2)//2], ye2[len(ye2)//2]), ...)

    # Configure plot
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Axial Distance (mm)")
    ax.set_ylabel("Radial Distance (mm)")
    ax.grid(True, linestyle=':', linewidth='0.5')
    ax.legend(fontsize=8)
    ax.minorticks_on()


# --- 3D Plotting Utilities ---

def ring(r, z_start, thickness, n_theta=30):
    """Generates coordinates for a 3D ring segment."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    # Create a single ring surface at z_start with some thickness
    # This plots a surface band, not just a line
    z = np.array([z_start, z_start + thickness])
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = r * np.cos(theta_grid)
    y = r * np.sin(theta_grid)
    return x, y, z_grid

def set_axes_equal_3d(ax: plt.Axes):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius]) # Use Z for axial direction

def plot3D(ax, contour, title="3D Bell Nozzle"):
    """Creates the 3D surface plot of the nozzle."""
    # Unpack contour data (only need upper half for surface of revolution)
    xe, ye, _, xe2, ye2, _, xbell, ybell, _ = contour

    # Combine x and y coordinates for the upper contour
    x_contour = np.concatenate((xe, xe2[1:], xbell[1:])) # Avoid duplicating connection points
    y_contour = np.concatenate((ye, ye2[1:], ybell[1:]))

    # Create the 3D surface by rotating the contour
    n_theta = 50 # Number of points for rotation
    theta = np.linspace(0, 2 * np.pi, n_theta)

    # Create meshgrid
    theta_grid, x_grid = np.meshgrid(theta, x_contour)
    # Interpolate radius values onto the x_grid
    radius_grid = np.interp(x_grid, x_contour, y_contour)

    # Convert cylindrical to Cartesian coordinates
    X = radius_grid * np.cos(theta_grid)
    Y = radius_grid * np.sin(theta_grid)
    Z = x_grid # Use Z as the axial direction for standard 3D plotting

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Configure plot
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Axial Distance Z (mm)") # Z is axial distance
    set_axes_equal_3d(ax) # Ensure proper aspect ratio
    # Optional: Set view angle
    ax.view_init(elev=20., azim=-60) # Adjust elevation and azimuth
    ax.grid(True)


# --- Combined Plotting Function ---

def plot(title, throat_radius_mm, angles, contour):
    """Creates a figure with both 2D and 3D plots of the bell nozzle."""
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title, fontsize=12, y=0.98)

    # 2D Plot
    ax1 = fig.add_subplot(121)
    plot_nozzle(ax1, "2D Contour & Dimensions", throat_radius_mm, angles, contour)

    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    plot3D(ax2, contour, "3D Surface")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    # plt.show() # Show should be called from the main script


# =============================================================
# CSV Export Function
# =============================================================

def export_contour_to_csv(contour_data, aratio, Rt_mm, filename="bell_nozzle_contour.csv"):
    """
    Exports the calculated nozzle contour (upper half) to a CSV file.

    Args:
        contour_data (tuple): Tuple containing the coordinate lists from bell_nozzle().
        aratio (float): Area ratio used for the calculation.
        Rt_mm (float): Throat radius (in mm) used for the calculation.
        filename (str): The name of the CSV file to create.

    Returns:
        bool: True if export was successful, False otherwise.
    """
    try:
        # Unpack contour data (only need upper half)
        xe, ye, _, xe2, ye2, _, xbell, ybell, _ = contour_data

        # Combine x and r coordinates for the upper contour, removing duplicates at junctions
        x_total = np.concatenate((xe, xe2[1:], xbell[1:]))
        r_total = np.concatenate((ye, ye2[1:], ybell[1:]))

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header information
            writer.writerow([f"Rao Bell Nozzle Contour"])
            writer.writerow([f"Area Ratio (Ae/At)", f"{aratio:.5f}"])
            writer.writerow([f"Throat Radius (Rt)", f"{Rt_mm:.3f} mm"])
            # writer.writerow([f"Length Percentage", f"{l_percent} %"]) # Need l_percent here
            # TODO: Pass l_percent to this function if header is desired
            writer.writerow([]) # Blank line
            writer.writerow(['x (mm)', 'r (mm)']) # Column headers

            # Write data points
            for x, r in zip(x_total, r_total):
                writer.writerow([f"{x:.4f}", f"{r:.4f}"]) # Use more precision maybe
        return True
    except IOError as e:
        print(f"Error writing CSV file '{filename}': {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during CSV export: {e}")
        return False

# =============================================================
# # Test Block (optional)
# # =============================================================
# if __name__ == "__main__":
#     print("Testing bell_nozzle_module...")

#     # Constants for testing (from original script)
#     k_test = 1.667
#     l_percent_test = 80
#     aratio_test = 12
#     throat_radius_test = 5 # mm

#     try:
#         # Calculate contour
#         angles_test, contour_test = bell_nozzle(k_test, aratio_test, throat_radius_test, l_percent_test)
#         print(f"Calculated angles (Ln_mm, theta_n_rad, theta_e_rad): {angles_test}")

#         # Plot
#         title_test = (f'TEST Bell Nozzle ({l_percent_test}% Rao)\n'
#                       f'Area Ratio = {aratio_test:.1f}, Throat Radius = {throat_radius_test:.1f} mm')
#         plot(title_test, throat_radius_test, angles_test, contour_test)
#         plt.show() # Show plot when testing module directly

#         # Export CSV
#         csv_success = export_contour_to_csv(contour_test, aratio_test, throat_radius_test,
#                                             f"test_bell_contour_ar{aratio_test}_rt{throat_radius_test}.csv")
#         if csv_success:
#             print("Test CSV export successful.")
#         else:
#             print("Test CSV export failed.")

#     except Exception as e:
#         print(f"Error during module test: {e}")