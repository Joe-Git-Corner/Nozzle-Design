import numpy as np
from scipy.optimize import fsolve

# =============================================================
# Helper Function Definitions (Isentropic Flow, Normal Shock)
# =============================================================

# Note: This isentropic_flow function is specific to this module's needs.
# It could potentially be merged with the one in nozzle_calculator.py
# into a shared utility module if desired.
def isentropic_flow(value, input_mode, gamma, output_mode):
    """
    Compute Mach number or area ratio for isentropic flow.
    Used internally by shock calculations.
    - value: Input value (A/A* or M)
    - input_mode: 'Asub' (subsonic A/A*), 'Asup' (supersonic A/A*), 'M' (Mach number)
    - gamma: Ratio of specific heats
    - output_mode: 'M' (return Mach), 'AAs' (return A/A*)
    """
    gm1 = gamma - 1
    gp1 = gamma + 1

    if output_mode == 'M' and input_mode in ['Asub', 'Asup']:
        A_Astar = value
        if A_Astar < 1.0: return np.nan # Physically impossible
        if np.isclose(A_Astar, 1.0): return 1.0

        def func(M):
            if M <= 0: return np.inf # Avoid division by zero or invalid M
            term2_base = 1 + (gm1 / 2) * M**2
            if term2_base <= 0: return np.inf # Avoid invalid base for power
            term2 = (2 / gp1) * term2_base
            exponent = gp1 / gm1
            val = (1 / M**2) * (term2**exponent)
            return A_Astar**2 - val

        if input_mode == 'Asub':
            M_guess = 0.5 if A_Astar < 5 else 1.0 / A_Astar # Initial guess for subsonic
            M, _, ier, _ = fsolve(func, M_guess, xtol=1e-7, full_output=True)
            if ier != 1 or M[0] >= 1.0: # Check success and validity
                 M_guess = 0.1 # Try another guess
                 M, _, ier, _ = fsolve(func, M_guess, xtol=1e-7, full_output=True)
                 if ier != 1 or M[0] >= 1.0: return np.nan # Failed
            return M[0]
        else: # Asup
            M_guess = 1.5 if A_Astar < 5 else A_Astar**0.5 # Initial guess for supersonic
            if M_guess <= 1.0: M_guess = 1.1
            M, _, ier, _ = fsolve(func, M_guess, xtol=1e-7, full_output=True)
            if ier != 1 or M[0] <= 1.0: # Check success and validity
                 M_guess = 5.0 # Try another guess
                 M, _, ier, _ = fsolve(func, M_guess, xtol=1e-7, full_output=True)
                 if ier != 1 or M[0] <= 1.0: return np.nan # Failed
            return M[0]

    elif output_mode == 'AAs' and input_mode == 'M':
        M = value
        if M <= 0: return np.nan
        if M == 1.0: return 1.0
        term2_base = 1 + (gm1 / 2) * M**2
        if term2_base <= 0: return np.nan
        term2 = (2 / gp1) * term2_base
        exponent = gp1 / gm1
        A_Astar = np.sqrt((1 / M**2) * (term2**exponent))
        return A_Astar
    else:
        raise ValueError("Invalid input/output mode combination for isentropic_flow")

def normal_shock(M1, input_type, gamma, output_type):
    """
    Compute downstream properties across a normal shock.
    - M1: Upstream Mach number (must be > 1)
    - input_type: 'M1'
    - gamma: Ratio of specific heats
    - output_type: 'M2' (downstream Mach), 'P2_P1' (static pressure ratio),
                   'P02_P01' (total pressure ratio), 'T2_T1' (static temp ratio)
    """
    if M1 <= 1.0:
        # Normal shock relations are not valid for M1 <= 1
        # Return identity values or NaN depending on expected behavior
        if output_type == 'M2': return M1
        if output_type in ['P2_P1', 'T2_T1']: return 1.0
        if output_type == 'P02_P01': return 1.0
        return np.nan # Default NaN for invalid input

    gm1 = gamma - 1
    gp1 = gamma + 1
    M1_sq = M1**2

    if input_type == 'M1':
        if output_type == 'M2':
            num = 1 + (gm1 / 2) * M1_sq
            den = gamma * M1_sq - (gm1 / 2)
            if den <= 0: return np.nan # Avoid sqrt of negative/zero
            M2 = np.sqrt(num / den)
            return M2
        elif output_type == 'P2_P1':
            P2_P1 = 1 + (2 * gamma / gp1) * (M1_sq - 1)
            return P2_P1
        elif output_type == 'P02_P01':
            term1_base = (gp1 * M1_sq) / (2 + gm1 * M1_sq)
            term2_base = (2 * gamma * M1_sq - gm1) / gp1
            if term1_base <= 0 or term2_base <= 0: return np.nan # Avoid invalid power base
            term1 = term1_base**(gamma / gm1)
            term2 = term2_base**(-1 / gm1)
            P02_P01 = term1 * term2
            # Add check for P0 loss (P02_P01 should be <= 1)
            return min(P02_P01, 1.0) # Cap at 1.0 due to potential numerical issues
        elif output_type == 'T2_T1':
            T2_T1 = (1 + (2 * gamma / gp1) * (M1_sq - 1)) * \
                    ((2 + gm1 * M1_sq) / (gp1 * M1_sq))
            return T2_T1
        else:
             raise ValueError(f"Invalid output_type '{output_type}' for normal_shock")
    else:
        raise ValueError(f"Invalid input_type '{input_type}' for normal_shock")

def ns_nozzle_calc_pb_p0(Arat, Ae_At, g):
    """
    Helper function for iterative solver.
    Computes the back-to-stagnation pressure ratio (Pb/P0) assuming a
    normal shock occurs at the location where the area ratio is Arat (A1/A*).

    Args:
        Arat (float): Area ratio A1/A* where the shock is assumed to occur.
        Ae_At (float): Exit-to-throat area ratio of the nozzle.
        g (float): Ratio of specific heats.

    Returns:
        float: Calculated Pb/P0 for the assumed shock location. Returns NaN on error.
    """
    gm1 = g - 1
    gp1 = g + 1
    gogm1 = g / gm1
    gm1o2 = gm1 / 2

    # Step 1: Supersonic Mach number (M1) just before the shock at Arat
    M1 = isentropic_flow(Arat, 'Asup', g, 'M')
    if np.isnan(M1): return np.nan # Error in isentropic calculation

    # Step 2: Mach number (M2) and total pressure ratio (P02/P01) across the shock
    M2 = normal_shock(M1, 'M1', g, 'M2')
    P02_P01 = normal_shock(M1, 'M1', g, 'P02_P01')
    if np.isnan(M2) or np.isnan(P02_P01): return np.nan # Error in shock calculation

    # Step 3: Effective sonic area ratio after shock (A*2 / A*1)
    # Note: A* is throat area A_t. A*1 is the reference throat area.
    # P01/P02 = A*2/A*1 relationship holds.
    Astar2_Astar1 = 1 / P02_P01 if P02_P01 != 0 else np.inf

    # Step 4: Subsonic area ratio from the shock location (A1=Arat*A*1) to the exit (Ae)
    # Ae / A*2 = (Ae / A*1) / (A*2 / A*1) = Ae_At / Astar2_Astar1
    Ae_Astar2 = Ae_At / Astar2_Astar1
    if Ae_Astar2 < 1.0:
        # This can happen if Ae_At is small and shock is strong (large P0 loss)
        # Physically means the flow would likely choke again downstream if area decreases
        # For a diverging nozzle, this indicates the flow remains subsonic M < 1
        # Let's assume M_exit will be subsonic corresponding to this Ae_Astar2
        pass # Proceed to calculate Me

    # Step 5: Exit Mach number (Me), which must be subsonic after the shock
    Me = isentropic_flow(Ae_Astar2, 'Asub', g, 'M')
    if np.isnan(Me): return np.nan # Error in isentropic calculation

    # Step 6: Exit pressure ratio (Pe/P02) and back pressure ratio (Pb/P01)
    # Assuming exit pressure Pe equals back pressure Pb
    # Pe/P02 = (1 + gm1o2 * Me^2)^(-g/gm1)
    Pe_P02_term_base = 1 + gm1o2 * Me**2
    if Pe_P02_term_base <= 0: return np.nan
    Pe_P02 = Pe_P02_term_base**(-gogm1)

    # Pb/P01 = (Pe/P02) * (P02/P01)
    Pb_P0 = Pe_P02 * P02_P01

    return Pb_P0

# =============================================================
# Main Calculation Function
# =============================================================

def calculate_normal_shock_location(Ae_At, g, Pb_P0_target):
    """
    Calculates the location (A1/A*) of a normal shock inside a CD nozzle,
    given the nozzle area ratio, gamma, and the target back pressure ratio.
    Also determines the pressure boundaries for different flow regimes.

    Args:
        Ae_At (float): Exit-to-throat area ratio (must be >= 1).
        g (float): Ratio of specific heats.
        Pb_P0_target (float): Target back-pressure to stagnation-pressure ratio.

    Returns:
        dict: A dictionary containing the results:
            'status' (str): Description of the flow regime.
            'Pe_P0_Sub' (float): Pb/P0 for subsonic isentropic flow at exit.
            'Pb_P0_Sup' (float): Pb/P0 for supersonic isentropic flow at exit (ideal expansion).
            'Pe_P0_NSE' (float): Pb/P0 for a normal shock exactly at the exit.
            'shock_in_nozzle' (bool): True if a shock is predicted inside.
            'shock_A_At_iter' (float/None): Shock location A1/A* from iterative method (or None).
            'shock_A_At_dir' (float/None): Shock location A1/A* from direct method (or None).
            'error_iter' (float/None): Final error in iterative method (or None).
    """
    results = {
        'status': 'Calculation Pending',
        'Pe_P0_Sub': None,
        'Pb_P0_Sup': None,
        'Pe_P0_NSE': None,
        'shock_in_nozzle': False,
        'shock_A_At_iter': None,
        'shock_A_At_dir': None,
        'error_iter': None
    }

    if Ae_At < 1.0:
        results['status'] = "Error: Ae/At must be >= 1."
        return results

    # --- Convenient parameters ---
    gm1 = g - 1
    gp1 = g + 1
    gogm1 = g / gm1
    gm1o2 = gm1 / 2

    # --- Calculate Pressure Regime Boundaries ---
    # 1. Fully Subsonic Isentropic Flow (Upper pressure boundary)
    Me_Sub = isentropic_flow(Ae_At, 'Asub', g, 'M')
    if np.isnan(Me_Sub):
        results['status'] = "Error calculating subsonic exit Mach."
        return results
    Pe_P0_Sub_base = 1 + gm1o2 * Me_Sub**2
    Pe_P0_Sub = Pe_P0_Sub_base**(-gogm1) if Pe_P0_Sub_base > 0 else np.nan
    results['Pe_P0_Sub'] = Pe_P0_Sub

    # 2. Fully Supersonic Isentropic Flow (Lower pressure boundary - ideal expansion)
    Me_Sup = isentropic_flow(Ae_At, 'Asup', g, 'M')
    if np.isnan(Me_Sup):
        # This can happen for very low Ae_At where supersonic flow isn't established
        results['status'] = "Warning: Could not calculate supersonic exit Mach (Ae/At might be too low)."
        results['Pb_P0_Sup'] = Pe_P0_Sub # Treat as if flow remains subsonic
        results['Pe_P0_NSE'] = Pe_P0_Sub
    else:
        Pb_P0_Sup_base = 1 + gm1o2 * Me_Sup**2
        Pb_P0_Sup = Pb_P0_Sup_base**(-gogm1) if Pb_P0_Sup_base > 0 else np.nan
        results['Pb_P0_Sup'] = Pb_P0_Sup

        # 3. Normal Shock Exactly at Nozzle Exit
        P2_P1_exit = normal_shock(Me_Sup, 'M1', g, 'P2_P1') # Pressure ratio across shock
        if np.isnan(P2_P1_exit) or np.isnan(Pb_P0_Sup):
             results['status'] = "Error calculating pressure ratio for shock at exit."
             results['Pe_P0_NSE'] = Pb_P0_Sup # Fallback value
        else:
            # Pe_P0_NSE = (P2/P1)_exit * (P1/P0)_sup = P2_P1_exit * Pb_P0_Sup
            Pe_P0_NSE = P2_P1_exit * Pb_P0_Sup
            results['Pe_P0_NSE'] = Pe_P0_NSE


    # --- Determine Flow Regime based on Pb_P0_target ---
    # Check for calculation errors in boundaries
    if any(v is None or np.isnan(v) for v in [Pe_P0_Sub, Pb_P0_Sup, Pe_P0_NSE]):
         results['status'] = "Error: Failed to calculate pressure boundaries."
         return results

    # Compare target pressure with boundaries
    if Pb_P0_target >= Pe_P0_Sub:
        results['status'] = "Flow is fully subsonic or underexpanded (isentropic)."
    elif Pb_P0_target < results.get('Pb_P0_Sup', -np.inf): # Use get for safety if Pb_P0_Sup failed
        results['status'] = "Flow is overexpanded (isentropic supersonic or oblique shocks outside)."
    elif Pb_P0_target == results.get('Pe_P0_NSE', np.nan):
        results['status'] = "Normal shock at nozzle exit."
    elif Pb_P0_target < results.get('Pe_P0_NSE', np.nan):
         results['status'] = "Flow is overexpanded (oblique shocks outside nozzle)."
    elif Pe_P0_Sub > Pb_P0_target > Pe_P0_NSE:
        # --- Normal Shock Inside Nozzle ---
        results['status'] = "Normal shock inside nozzle."
        results['shock_in_nozzle'] = True

        # Method 1: Iterative Solver (Bisection)
        errTol = 1e-7 # Error tolerance for Pb/P0 match
        maxIter = 100 # Max iterations
        Arat_Lo = 1.00001 # Shock cannot be exactly at throat
        Arat_Hi = Ae_At
        err = float('inf')
        iter_count = 0
        Arat_Mid = (Arat_Lo + Arat_Hi) / 2 # Initialize

        while err > errTol and iter_count < maxIter:
            iter_count += 1
            Arat_Mid = 0.5 * (Arat_Hi + Arat_Lo)

            Pb_P0_Lo = ns_nozzle_calc_pb_p0(Arat_Lo, Ae_At, g)
            Pb_P0_Mid = ns_nozzle_calc_pb_p0(Arat_Mid, Ae_At, g)
            
            # Handle potential NaN results from the helper function
            if np.isnan(Pb_P0_Lo) or np.isnan(Pb_P0_Mid):
                 results['status'] = "Error during iterative solve (NaN encountered)."
                 Arat_Mid = np.nan # Mark as error
                 break # Exit loop on error

            errLo = Pb_P0_target - Pb_P0_Lo
            errMid = Pb_P0_target - Pb_P0_Mid

            if errLo * errMid < 0: # Root is between Lo and Mid
                Arat_Hi = Arat_Mid
            else: # Root is between Mid and Hi (or exactly at Mid/Hi)
                Arat_Lo = Arat_Mid

            err = abs(Pb_P0_Mid - Pb_P0_target) # Update error based on target pressure match
            # Alternate error: err = abs(Arat_Hi - Arat_Lo) # Based on interval size

        if iter_count >= maxIter:
             print(f"Warning: Iterative solver reached max iterations ({maxIter}). Result might be inaccurate.")
             
        if not np.isnan(Arat_Mid):
             results['shock_A_At_iter'] = Arat_Mid
             results['error_iter'] = err


        # Method 2: Direct Calculation (from original script logic)
        # This involves solving for exit Mach assuming subsonic flow after shock
        # then back-calculating P02/P01 and M1.
        try:
            # Calculate effective exit Mach (subsonic) based on Pb_P0_target and Ae_At
            # This requires solving Pb/P0 = (Pe/P02)*(P02/P01) where Pe/P02 relates to Me
            # and P02/P01 relates to M1. This seems complex to solve directly for M1.
            # The original script's "Direct Method" seems to solve a complex equation for M1
            # derived from relating P02/P01 to M1 and substituting into the overall pressure relation.
            
            # Let's try to replicate the final fsolve part from original:
            # First, estimate P02/P01 needed. Assume flow expands isentropically from P02 to Pe=Pb
            # So, Pb/P02 = (1+gm1o2*Me^2)^(-gogm1). We need Me.
            # Me is related to Ae/A*2 = (Ae/A*1)/(A*2/A*1) = Ae_At * P02_P01
            # This leads to coupled equations.

            # Revisit original direct method's terms: It calculates Me first using a formula.
            # This formula seems derived assuming the exit flow *is* isentropic from P02
            # and somehow relates Pb/P0 directly to Me without initially knowing P02/P01.
            # Let's use the formula given in the original script to find Me:
            term1 = -(1/gm1)
            term2 = 1/(gm1**2)
            term3 = 2/gm1
            term4 = (2/gp1)**(gp1/gm1) # (A*/A_e_isen_sub)^2 * (Pe/P0_isen_sub)^2 ? Check derivation
            term5_base = Pb_P0_target * Ae_At
            if term5_base == 0: raise ValueError("Pb_P0_target * Ae_At is zero")
            term5 = term5_base**-2
            
            sqrt_inner = term2 + term3 * term4 * term5
            if sqrt_inner < 0: raise ValueError("Negative value in sqrt for direct Me calc")
            
            Me_dir = np.sqrt(term1 + np.sqrt(sqrt_inner)) # Exit Mach assuming isentropic from P02

            # Now find P02/P01 needed to achieve this Me and Pb_P0_target
            # Pb/P01 = (Pb/P02) * (P02/P01)
            # Pb/P02 = (1 + gm1o2*Me_dir^2)^(-gogm1)
            P0e_Pe_base = 1 + gm1o2 * Me_dir**2
            if P0e_Pe_base <= 0: raise ValueError("Invalid base for P0e_Pe calc")
            P0e_Pe = P0e_Pe_base**(g/gm1) # This is P02/Pe or P02/Pb
            
            P02_P01_needed = P0e_Pe * Pb_P0_target

            # Now find M1 that gives this P02/P01 ratio using normal shock relation
            def shock_p0_eq(M):
                p02_p01_calc = normal_shock(M, 'M1', g, 'P02_P01')
                # Ensure calculation succeeded
                if np.isnan(p02_p01_calc): return np.inf
                return p02_p01_calc - P02_P01_needed

            # Solve for M1 (must be > 1)
            M1_guess = 1.5 # Start guess
            M1_dir, _, ier, _ = fsolve(shock_p0_eq, M1_guess, xtol=1e-7, full_output=True)
            if ier != 1 or M1_dir[0] <= 1.0:
                 M1_guess = 3.0 # Try another guess
                 M1_dir, _, ier, _ = fsolve(shock_p0_eq, M1_guess, xtol=1e-7, full_output=True)
                 if ier != 1 or M1_dir[0] <= 1.0:
                     raise ValueError("Direct method failed to find valid M1 for shock.")

            M1_dir = M1_dir[0]

            # Finally, find the area ratio corresponding to this M1
            ANS_A1s_Dir = isentropic_flow(M1_dir, 'M', g, 'AAs')
            results['shock_A_At_dir'] = ANS_A1s_Dir

        except (ValueError, FloatingPointError) as e:
             print(f"Warning: Direct method calculation failed: {e}")
             results['shock_A_At_dir'] = None # Indicate failure

    else:
        # Status already set for other regimes
        pass

    return results


# =============================================================
# Test Block (optional)
# =============================================================
if __name__ == "__main__":
    print("Testing shock_calculator module...")

    # --- Test Case 1: Shock Inside Nozzle ---
    print("\n--- Test Case 1: Shock Expected Inside ---")
    Ae_At_test1 = 3.0 # Original example Ae/At
    g_test1 = 1.4     # Original example gamma
    # Select a Pb/P0 between NSE and Subsonic boundaries for Ae/At=3, g=1.4
    # Boundaries for Ae/At=3, g=1.4 are roughly: Sub=0.97, NSE=0.58, Sup=0.06
    Pb_P0_test1 = 0.7   # Should result in shock inside

    results1 = calculate_normal_shock_location(Ae_At_test1, g_test1, Pb_P0_test1)

    print(f"Target Pb/P0: {Pb_P0_test1}")
    print(f"Status: {results1['status']}")
    print(f"Boundaries: Sub={results1['Pe_P0_Sub']:.4f}, NSE={results1['Pe_P0_NSE']:.4f}, Sup={results1['Pb_P0_Sup']:.4f}")
    print(f"Shock in Nozzle? {results1['shock_in_nozzle']}")
    if results1['shock_in_nozzle']:
        print(f"Iterative A1/A*: {results1['shock_A_At_iter']:.5f} (Err: {results1['error_iter']:.2e})")
        print(f"Direct A1/A* : {results1['shock_A_At_dir']:.5f}")


    # --- Test Case 2: Overexpanded ---
    print("\n--- Test Case 2: Overexpanded ---")
    Ae_At_test2 = 11.18679 # From NozzleDesignCalc result with P_ambient=0.01333
    g_test2 = 1.667      # Argon
    # Use P_ambient / P0 from nozzle calc inputs
    Pb_P0_test2 = 0.01333 / 666.612 # ~ 2e-5

    results2 = calculate_normal_shock_location(Ae_At_test2, g_test2, Pb_P0_test2)

    print(f"Target Pb/P0: {Pb_P0_test2:.5g}")
    print(f"Status: {results2['status']}")
    print(f"Boundaries: Sub={results2['Pe_P0_Sub']:.5g}, NSE={results2['Pe_P0_NSE']:.5g}, Sup={results2['Pb_P0_Sup']:.5g}")
    print(f"Shock in Nozzle? {results2['shock_in_nozzle']}")

     # --- Test Case 3: Underexpanded / Subsonic ---
    print("\n--- Test Case 3: Underexpanded/Subsonic ---")
    Ae_At_test3 = 2.0
    g_test3 = 1.4
    Pb_P0_test3 = 0.95 # Higher than subsonic boundary

    results3 = calculate_normal_shock_location(Ae_At_test3, g_test3, Pb_P0_test3)

    print(f"Target Pb/P0: {Pb_P0_test3:.4f}")
    print(f"Status: {results3['status']}")
    print(f"Boundaries: Sub={results3['Pe_P0_Sub']:.4f}, NSE={results3['Pe_P0_NSE']:.4f}, Sup={results3['Pb_P0_Sup']:.4f}")
    print(f"Shock in Nozzle? {results3['shock_in_nozzle']}")