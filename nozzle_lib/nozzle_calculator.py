import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# =============================================================
# Helper Function Definitions (Isentropic Flow etc.)
# =============================================================

def get_area_ratio(M, gamma):
    """Calculate the area ratio (A/A*) for a given Mach number and specific heat ratio."""
    if M == 0: return np.inf # Handle M=0 case
    if M == 1: return 1.0
    term1 = 1 / M
    term2 = (2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return term1 * term2**exponent

def get_pressure(P0, M, gamma):
    """Calculate static pressure given stagnation pressure, Mach number, and gamma."""
    return P0 / (1 + 0.5 * (gamma - 1) * M**2)**(gamma / (gamma - 1))

def get_temperature(T0, M, gamma):
    """Calculate static temperature given stagnation temperature, Mach number, and gamma."""
    return T0 / (1 + 0.5 * (gamma - 1) * M**2)

def get_mach_from_area_ratio(A_Astar, gamma, supersonic=False):
    """Calculate Mach number from area ratio using the isentropic relation."""
    gm1 = gamma - 1
    gp1 = gamma + 1
    
    if np.isclose(A_Astar, 1.0, atol=1e-6):
        return 1.0
    if A_Astar < 1.0:
        # Physical impossibility for steady isentropic flow
        print(f"Warning: A/A* = {A_Astar} < 1 requested in get_mach_from_area_ratio. Returning NaN.")
        return np.nan

    def func(M):
        # Check for M=0 during solve which can cause division by zero
        if M == 0: return A_Astar**2 - np.inf
        term2 = (2 / gp1) * (1 + (gm1 / 2) * M**2)
        # Prevent overflow/invalid value in power for very large M or specific gamma
        if term2 <= 0: return np.inf
        exponent = gp1 / gm1
        val = (1 / M**2) * (term2**exponent)
        return A_Astar**2 - val

    if supersonic:
        # Improved guess for supersonic flow
        M_guess = 1.0 + 2.0 * (A_Astar - 1.0) # Linear approx near M=1
        if A_Astar > 5: M_guess = A_Astar**0.5 # Asymptotic behavior
        if M_guess <= 1.0: M_guess = 1.1 # Ensure guess > 1

        M, infodict, ier, mesg = fsolve(func, M_guess, xtol=1e-7, full_output=True)
        if ier != 1 or M[0] <= 1.0: # Check if fsolve succeeded and result is supersonic
            # Try a different guess if the first one failed or gave subsonic
            M_guess = 5.0
            M, infodict, ier, mesg = fsolve(func, M_guess, xtol=1e-7, full_output=True)
            if ier != 1 or M[0] <= 1.0:
                print(f"Warning: Supersonic solver failed for A/A*={A_Astar}. Msg: {mesg}. Returning NaN.")
                return np.nan
        return M[0]
    else:
        # Improved guess for subsonic flow
        # Approximate relation for small M: A/A* ≈ 1/M
        M_guess = 1.0 / A_Astar if A_Astar > 1.5 else 0.5
        if M_guess >= 1.0: M_guess = 0.9 # Ensure guess < 1

        M, infodict, ier, mesg = fsolve(func, M_guess, xtol=1e-7, full_output=True)
        if ier != 1 or M[0] >= 1.0 or M[0] <= 0: # Check if fsolve succeeded and result is valid subsonic
             # Try a different guess
            M_guess = 0.1
            M, infodict, ier, mesg = fsolve(func, M_guess, xtol=1e-7, full_output=True)
            if ier != 1 or M[0] >= 1.0 or M[0] <= 0:
                print(f"Warning: Subsonic solver failed for A/A*={A_Astar}. Msg: {mesg}. Returning NaN.")
                return np.nan
        return M[0]


# =============================================================
# Main Calculation Function
# =============================================================

def calculate_nozzle_dimensions_and_performance(inputs):
    """
    Calculates nozzle dimensions (conical approximation for performance)
    and performance metrics.

    Args:
        inputs (dict): Dictionary containing all required input parameters:
            P0, T0, P_ambient, R_throat, R_chamber, theta_conv, theta_div,
            gamma, R_specific, g, pressure_tolerance, M_min, M_max, num_points

    Returns:
        dict: Dictionary containing calculated results, including:
            M_exit, P_exit, area_ratio_exit, A_exit, R_exit, L_div, L_conv,
            v_exit, mass_flow_rate, thrust, C_F, c_star, I_sp,
            M_range, p_range, T_range, x_total, P_total, x_total_mm, A_Astar_x
            (and others for potential plotting)
    """
    # Unpack inputs
    P0 = inputs['P0']
    T0 = inputs['T0']
    P_ambient = inputs['P_ambient']
    R_throat = inputs['R_throat']
    R_chamber = inputs['R_chamber']
    theta_conv = inputs['theta_conv']
    theta_div = inputs['theta_div']
    gamma = inputs['gamma']
    R_specific = inputs['R_specific']
    g = inputs['g']
    pressure_tolerance = inputs['pressure_tolerance']
    M_min = inputs['M_min']
    M_max = inputs['M_max']
    num_points = inputs['num_points']

    # --- Calculations ---
    A_throat = np.pi * R_throat**2  # Throat area [m^2]
    M_range = np.linspace(M_min, M_max, num_points) # Mach range

    # Compute pressure and temperature across Mach range
    p_range = get_pressure(P0, M_range, gamma)
    T_range = get_temperature(T0, M_range, gamma)

    # Find optimal Mach number where exit pressure matches ambient (within tolerance)
    # This simulates designing for optimal expansion to P_ambient
    target_P_exit = P_ambient
    diff = np.abs(p_range - target_P_exit)
    idx = np.argmin(diff)
    
    # Check if the minimum difference is within tolerance
    if diff[idx] <= pressure_tolerance:
        M_exit = M_range[idx]
        P_exit = p_range[idx]
        print(f"Target ambient pressure {target_P_exit:.5f} Pa matched within tolerance.")
    else:
        # If no point is close enough, it implies either under/over expansion significantly
        # We need to decide the design goal. Original code searched for first P <= P_ambient + tol.
        # Let's stick to the closest match found, but issue a warning.
        M_exit = M_range[idx]
        P_exit = p_range[idx]
        print(f"Warning: Could not match P_ambient ({target_P_exit:.5f} Pa) within tolerance {pressure_tolerance} Pa.")
        print(f"         Closest pressure found is {P_exit:.5f} Pa at M={M_exit:.3f}.")
        # Alternative based on original logic: find first index below threshold
        # idx_orig = np.where(p_range <= P_ambient + pressure_tolerance)[0]
        # if len(idx_orig) > 0:
        #     i = idx_orig[0]
        #     M_exit = M_range[i]
        #     P_exit = p_range[i]
        # else:
        #     print("No suitable Mach number found within range using original logic. Using maximum Mach.")
        #     M_exit = M_range[-1]
        #     P_exit = p_range[-1]
            
    area_ratio_exit = get_area_ratio(M_exit, gamma)

    # Verify supersonic flow condition based on pressure ratio Pr=P*/P0
    P_choked_limit = get_pressure(P0, 1.0, gamma)
    if P_exit > P_choked_limit:
        print(f"Warning: Calculated exit pressure {P_exit:.3f} Pa is higher than choked pressure limit {P_choked_limit:.3f} Pa.")
        print("         This suggests the flow might not reach supersonic speeds or the target P_ambient is too high.")
        # Potentially adjust M_exit back towards 1 or handle as an error depending on requirements.
        # For now, we proceed with the calculated M_exit based on pressure matching.

    # Nozzle geometry calculations (Conical Approximation)
    A_exit = A_throat * area_ratio_exit
    R_exit = np.sqrt(A_exit / np.pi)
    L_div = (R_exit - R_throat) / np.tan(theta_div) if np.tan(theta_div) != 0 else 0 # Diverging length [m]
    L_conv = (R_chamber - R_throat) / np.tan(theta_conv) if np.tan(theta_conv) != 0 else 0 # Converging length [m]

    # Nozzle profile along diverging section (Conical)
    x_nozzle = np.linspace(0, L_div, 100)  # [m] Starts at throat
    r_x_div = R_throat + x_nozzle * np.tan(theta_div)
    A_x_div = np.pi * r_x_div**2
    A_Astar_x_div = A_x_div / A_throat
    # Vectorized calculation of Mach number along the diverging section
    M_x_div = np.array([get_mach_from_area_ratio(a, gamma, supersonic=True) for a in A_Astar_x_div])
    P_x_div = get_pressure(P0, M_x_div, gamma)

    # Converging section calculations (Conical)
    x_conv = np.linspace(-L_conv, 0, 100) # [m] Ends at throat
    # Ensure R_chamber >= R_throat for sensible geometry
    if R_chamber < R_throat:
         print("Warning: R_chamber is less than R_throat. Converging section calculation might be invalid.")
         r_conv = np.linspace(R_chamber, R_throat, 100) # Simple linear taper if invalid
    elif L_conv > 0:
         r_conv = R_chamber + (x_conv + L_conv) * (R_throat - R_chamber) / L_conv
    else: # Handle case L_conv = 0 (straight connection to throat)
         r_conv = np.full_like(x_conv, R_throat)
         
    A_conv = np.pi * r_conv**2
    A_Astar_conv = A_conv / A_throat
    # Vectorized calculation of Mach number along the converging section
    M_conv = np.array([get_mach_from_area_ratio(a, gamma, supersonic=False) for a in A_Astar_conv])
    # Handle potential NaN results from get_mach_from_area_ratio
    M_conv = np.nan_to_num(M_conv, nan=0.0) # Replace NaN with 0 or another appropriate value
    P_conv = get_pressure(P0, M_conv, gamma)


    # Combine sections for plotting pressure profile
    # Avoid duplicating the throat point (x=0)
    x_total = np.concatenate((x_conv[:-1], x_nozzle))
    P_total = np.concatenate((P_conv[:-1], P_x_div))
    r_total = np.concatenate((r_conv[:-1], r_x_div)) # Combine radius profile
    M_total = np.concatenate((M_conv[:-1], M_x_div)) # Combine Mach profile
    x_total_mm = x_total * 1000
    L_div_mm = L_div * 1000

    # Performance calculations
    # Exit velocity - handle potential complex numbers if P_exit > P0
    pressure_ratio_term = P_exit / P0
    if pressure_ratio_term > 1:
        print(f"Warning: P_exit/P0 = {pressure_ratio_term:.3f} > 1. Cannot calculate real exit velocity.")
        v_exit = 0
    else:
        v_exit = math.sqrt(max(0, 2 * gamma / (gamma - 1) * R_specific * T0 * (1 - pressure_ratio_term**((gamma - 1) / gamma))))

    # Mass flow rate (choked flow assumption)
    try:
        mass_flow_rate = A_throat * P0 * math.sqrt(gamma * (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)) / (R_specific * T0))
    except ValueError:
         print("Warning: Potential math domain error in mass flow rate calculation. Check inputs.")
         mass_flow_rate = 0

    # Thrust
    # Using P_exit calculated based on matching P_ambient (optimal expansion assumption)
    # If P_exit couldn't match P_ambient, this thrust is for the P_exit achieved.
    # For true thrust with mismatched pressure, delta_P should use P_ambient.
    # Let's calculate both for clarity.
    delta_P_exit_ambient = P_exit - P_ambient
    thrust_matched_exit = mass_flow_rate * v_exit + delta_P_exit_ambient * A_exit
    
    # Thrust assuming flow expands to calculated P_exit, but ambient is P_ambient
    thrust_actual_ambient = mass_flow_rate * v_exit + (P_exit - P_ambient) * A_exit


    # C_F (Thrust Coefficient) - typically based on thrust_actual_ambient
    C_F = thrust_actual_ambient / (P0 * A_throat) if (P0 * A_throat) != 0 else 0

    # c_star (Characteristic Velocity)
    c_star = P0 * A_throat / mass_flow_rate if mass_flow_rate != 0 else 0

    # I_sp (Specific Impulse)
    I_sp = thrust_actual_ambient / (mass_flow_rate * g) if (mass_flow_rate * g) != 0 else 0
    # Alternate definition using C_F and c*
    # I_sp_alt = c_star * C_F / g if g != 0 else 0


    # Store results in a dictionary
    results = {
        'M_exit': M_exit,
        'P_exit': P_exit, # Pressure at exit based on calculation target
        'area_ratio_exit': area_ratio_exit,
        'A_throat': A_throat,
        'A_exit': A_exit,
        'R_exit': R_exit,
        'L_div': L_div,
        'L_div_mm': L_div_mm,
        'L_conv': L_conv,
        'v_exit': v_exit,
        'mass_flow_rate': mass_flow_rate,
        'thrust': thrust_actual_ambient, # Thrust considering ambient pressure
        'thrust_matched_exit': thrust_matched_exit, # Thrust if P_exit=P_ambient achieved
        'C_F': C_F,
        'c_star': c_star,
        'I_sp': I_sp,
        'gamma': gamma,
        'P0': P0,
        'T0': T0,
        'P_ambient': P_ambient, 
        
        # Data for plotting
        'M_range': M_range,
        'p_range': p_range,
        'T_range': T_range,
        'x_total': x_total,
        'P_total': P_total,
        'r_total': r_total,
        'M_total': M_total,
        'x_total_mm': x_total_mm,
        'A_Astar_x_div': A_Astar_x_div, # Area ratio along diverging section
        'x_nozzle': x_nozzle # x-coordinates for diverging section
    }

    return results

# =============================================================
# Plotting Function (Optional - can be called from main)
# =============================================================

def plot_nozzle_characteristics(results):
    """
    Generates plots based on the results from calculate_nozzle_dimensions_and_performance.

    Args:
        results (dict): Dictionary containing the calculation results.
    """
    # Unpack necessary results
    M_range = results['M_range']
    p_range = results['p_range']
    T_range = results['T_range']
    gamma = results['gamma']
    M_exit = results['M_exit']
    P_exit = results['P_exit']
    area_ratio_exit = results['area_ratio_exit']
    L_div_mm = results['L_div_mm']
    x_total_mm = results['x_total_mm']
    P_total = results['P_total']
    
    # --- Plotting ---
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Nozzle Characteristics (Isentropic Flow, Conical Approx.)")

    # Area ratio vs Mach number
    area_ratios = np.array([get_area_ratio(M, gamma) for M in M_range])
    ax[0, 0].plot(M_range, area_ratios, label='A/A*')
    ax[0, 0].plot(M_exit, area_ratio_exit, 'ro', label=f'Exit M={M_exit:.2f}, A/A*={area_ratio_exit:.2f}')
    ax[0, 0].set_title("Area Ratio vs Mach Number")
    ax[0, 0].set_xlabel("Mach Number")
    ax[0, 0].set_ylabel("A/A*")
    ax[0, 0].set_ylim(bottom=0) # Ensure y-axis starts at 0
    ax[0, 0].grid(True)
    ax[0, 0].legend()

    # Pressure vs Mach number
    ax[0, 1].plot(M_range, p_range, label='Pressure')
    ax[0, 1].plot(M_exit, P_exit, 'ro', label=f'Exit M={M_exit:.2f}, P={P_exit:.2f} Pa')
    ax[0, 1].set_title("Pressure vs Mach Number")
    ax[0, 1].set_xlabel("Mach Number")
    ax[0, 1].set_ylabel("Pressure (Pa)")
    ax[0, 1].set_yscale('log') # Use log scale for pressure often
    ax[0, 1].grid(True, which='both')
    ax[0, 1].legend()

    # Temperature vs Mach number
    ax[1, 0].plot(M_range, T_range, label='Temperature')
    T_exit = get_temperature(results['T0'], M_exit, gamma)
    ax[1, 0].plot(M_exit, T_exit, 'ro', label=f'Exit M={M_exit:.2f}, T={T_exit:.1f} K')
    ax[1, 0].set_title("Temperature vs Mach Number")
    ax[1, 0].set_xlabel("Mach Number")
    ax[1, 0].set_ylabel("Temperature (K)")
    ax[1, 0].set_ylim(bottom=0)
    ax[1, 0].grid(True)
    ax[1, 0].legend()

    # Pressure vs distance along nozzle
    # Filter data if needed (e.g., remove initial drop if calculation issue exists)
    # mask = x_total_mm >= -10 # Example filter
    # x_plot = x_total_mm[mask]
    # P_plot = P_total[mask]
    x_plot = x_total_mm
    P_plot = P_total

    ax[1, 1].plot(x_plot, P_plot, label='Static Pressure')
    ax[1, 1].axvline(x=0, color='k', linestyle='--', linewidth=0.8, label='Throat')
    ax[1, 1].axhline(y=results['P_ambient'], color='r', linestyle=':', linewidth=0.8, label='Ambient Pressure')
    ax[1, 1].plot(L_div_mm, P_exit, 'go', label=f'Exit P={P_exit:.2f} Pa')
    ax[1, 1].set_title(f"Pressure Profile (Conical Approx. L_div={L_div_mm:.2f} mm)")
    ax[1, 1].set_xlabel("Distance from Throat (mm)")
    ax[1, 1].set_ylabel("Pressure (Pa)")
    ax[1, 1].set_yscale('log') # Use log scale for pressure often
    ax[1, 1].grid(True, which='both')
    ax[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap


# =============================================================
# Test Block (optional)
# =============================================================
# if __name__ == "__main__":
#     # Example usage for testing the module directly
#     print("Testing nozzle_calculator module...")

#     # Define some test inputs (matching original script)
#     test_inputs = {
#         'P0': 666.612,
#         'T0': 300,
#         'P_ambient': 0.01333,
#         'R_throat': 0.005,
#         'R_chamber': 0.08,
#         'theta_conv': np.deg2rad(60),
#         'theta_div': np.deg2rad(15),
#         'gamma': 1.667,
#         'R_specific': 208.13, # Calculated for Argon
#         'g': 9.8,
#         'pressure_tolerance': 1.0,
#         'M_min': 0.1,
#         'M_max': 10,
#         'num_points': 1000
#     }

#     try:
#         results = calculate_nozzle_dimensions_and_performance(test_inputs)
        
#         print("\n--- Test Calculation Results ---")
#         print(f"Selected exit Mach number: {results['M_exit']:.3f}")
#         print(f"Exit pressure: {results['P_exit']:.5f} Pa")
#         print(f"Area ratio (A_exit/A_throat): {results['area_ratio_exit']:.5f}")
#         print(f"Exit area: {results['A_exit']:.8f} m²")
#         print(f"Exit radius: {results['R_exit']:.8f} m")
#         print(f"Diverging length: {results['L_div_mm']:.2f} mm")
#         print(f"Exit velocity: {results['v_exit']:.2f} m/s")
#         print(f"Mass flow rate: {results['mass_flow_rate']:.5f} kg/s")
#         print(f"Thrust: {results['thrust']:.5f} N")
#         print(f"Thrust coefficient (C_F): {results['C_F']:.5f}")
#         print(f"Characteristic velocity (c*): {results['c_star']:.2f} m/s")
#         print(f"Specific impulse (I_sp): {results['I_sp']:.2f} s")

#         # Test plotting
#         plot_nozzle_characteristics(results)
#         plt.show()

#     except Exception as e:
#         print(f"Error during module test: {e}")