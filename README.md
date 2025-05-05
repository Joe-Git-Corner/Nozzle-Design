# Python Nozzle Design and Analysis Toolkit

## Overview

This project provides a set of Python scripts for the preliminary design and analysis of rocket nozzles. It calculates key performance metrics, determines geometric properties for conical and Rao bell nozzles, checks for normal shock conditions within the nozzle, and allows for parameter sweeps to explore design trade-offs.

The toolkit currently focuses on:
* **Isentropic Flow Calculations:** Determining pressure, temperature, and Mach number relationships.
* **Conical Nozzle Performance:** Calculating thrust, specific impulse (Isp), mass flow rate, etc., based on a conical approximation for optimal expansion.
* **Rao Bell Nozzle Geometry:** Generating the contour coordinates for a thrust-optimized parabolic (Rao) bell nozzle based on empirical data and specified length percentage.
* **Normal Shock Check:** Determining the flow regime (subsonic, supersonic, shock in nozzle, shock at exit) based on the nozzle area ratio and back pressure ratio.
* **Visualization:** Plotting key performance characteristics and generating 2D/3D plots of the bell nozzle contour.
* **Data Export:** Saving bell nozzle contour coordinates to CSV files.
* **Parameter Sweeps:** Iterating over ranges of input parameters (like stagnation temperature and throat radius) to study their impact on performance and geometry.

## Features

* Calculates nozzle exit conditions (Mach, Pressure, Area Ratio) for optimal expansion (isentropic flow assumptions).
* Computes standard performance metrics (Thrust, Isp, C_F, c*).
* Generates 2D contour coordinates and 3D surface plots for Rao bell nozzles (60%, 80%, 90% length options).
* Calculates pressure boundaries for different flow regimes.
* Checks for the potential location of a normal shock within the nozzle based on the ambient pressure.
* Determines the required exit area ratio (Ae/At) for optimal expansion to a specified ambient pressure.
* Calculates the contour of an 80% Rao optimized bell nozzle for the determined area ratio.
* Exports bell nozzle contour data to CSV.
* Modular code structure for clarity and potential extension.

* Generates plots for:
    * **Single Run (Mode 1):**
        * Bell Nozzle Contour (2D and 3D)
        * Isentropic Flow Properties vs. Mach Number (Pressure Ratio, Temperature Ratio, Area Ratio)
        * Pressure Profile along Nozzle Axis (if contour calculated)
    * **Parameter Sweep (Mode 2):**
        * Isp vs. Stagnation Temperature (T0)
        * Thrust vs. T0
        * Mass Flow Rate vs. T0
        * Bell Nozzle Length vs. T0
* Performs parameter sweeps over Stagnation Temperature (T0) and Throat Radius (R_throat).
* Saves sweep results to a CSV file (`sweep_results.csv`).

## Inputs

Primary inputs are defined near the top of `main_nozzle_analysis.py`:

* **Physical Constants**: Molar Mass, Gamma, R_universal, Gravity.
* **Gas State Conditions**: P0 (Stagnation Pressure), T0 (Stagnation Temperature), P_AMBIENT (Ambient Pressure).
* **Nozzle Geometry**: R_THROAT_M (Throat Radius), R_CHAMBER_M (Chamber Radius), THETA_CONV_DEG (Converging Angle), THETA_DIV_DEG (Diverging Angle - *used for initial conical perf. approx.*).
* **Bell Nozzle Specific**: BELL_LENGTH_PERCENT (e.g., 80 for 80% Rao).
* **Calculation Parameters**: PRESSURE_TOLERANCE (for matching P_exit to P_ambient).
* **Sweep Parameters**: Defined within the `run_sweep_calculation` function (T0 range, R_throat range, steps, output control).


## Workflow & Algorithm

The main script (`main_nozzle_analysis.py`) orchestrates the analysis based on the following workflow (visualized in `Nozzle Calc workflow.jpg`):

1.  **Inputs:** Define primary inputs (stagnation conditions P0, T0; ambient pressure P_ambient; throat radius R_throat; material properties gamma, M_molar; geometry parameters R_chamber, angles, bell length %). Parameter ranges for T0 and R_throat can also be defined for sweeps.
2.  **Nozzle Performance Calculation (`nozzle_calculator.py`):**
    * Calculates the required exit area ratio (Ae/At) for optimal expansion to P_ambient using isentropic flow relations.
    * Determines the corresponding exit Mach number (Me) and exit pressure (Pe).
    * Calculates performance metrics (v_exit, mass flow, thrust, Isp, C_F, c*) based on these ideal conditions and a conical nozzle approximation.
3.  **Normal Shock Check (`shock_calculator.py`):**
    * Calculates the critical back pressure ratios defining flow regimes (fully subsonic, normal shock at exit, fully supersonic).
    * Compares the actual back pressure ratio (P_ambient / P0) to these boundaries.
    * If the pressure ratio indicates a normal shock *inside* the nozzle, it calculates the shock location (A/A*) using iterative and direct methods.
    * Reports the flow status. The main workflow may stop or issue warnings if a shock is detected inside, depending on design goals.
4.  **Bell Nozzle Geometry (`bell_nozzle_module.py`):**
    * If the shock check passes (or is acceptable), this module calculates the 2D contour of a Rao bell nozzle.
    * It uses the area ratio (Ae/At) from the performance calculation, the throat radius (Rt), specific heat ratio (gamma/k), and the desired length percentage (e.g., 80%).
    * It determines the nozzle length (Ln) and key angles (theta_n, theta_e) based on empirical data and interpolation.
    * The contour is generated using circular arcs for the throat region and a quadratic Bezier curve for the bell section, following Rao's method.
5.  **Output & Visualization:**
    * Results (performance metrics, shock status, dimensions) are printed to the console.
    * If calculated, the bell nozzle contour is plotted in 2D (with dimensions) and 3D.
    * Bell nozzle contour points are saved to a CSV file.
    * If a parameter sweep is run, trend plots summarizing the results are generated.

## File Structure
```
./
│
├── main_nozzle_analysis.py     # Main script to run analysis & sweeps
├── Nozzle Calc workflow.jpg    # Workflow diagram image
├── sweep_results.csv           # Optional: Output from parameter sweep
├── README.md                   # This file
│
└── nozzle_lib/                 # Subdirectory for calculation modules
    ├── init.py                 # Makes nozzle_lib a Python package
    ├── nozzle_calculator.py    # Isentropic flow, conical performance calcs & plots
    ├── bell_nozzle_module.py   # Rao bell nozzle contour, plotting & CSV export
    └── shock_calculator.py     # Normal shock location calculations
    └── output/                 # Optional: Directory for CSV output files
    └── plots/                  # Optional: Directory for saved plot images
```


## Modules Description

* **`main_nozzle_analysis.py`**: Entry point. Defines inputs, orchestrates calls to modules, handles parameter sweeps, prints summaries, and manages overall workflow.
* **`nozzle_lib/nozzle_calculator.py`**: Contains functions for 1D isentropic flow calculations and nozzle performance metrics based on a conical nozzle approximation. Includes an optional function to plot flow properties.
* **`nozzle_lib/bell_nozzle_module.py`**: Implements the Rao method for generating thrust-optimized parabolic bell nozzle contours. Provides functions for 2D and 3D plotting and CSV export of the contour.
* **`nozzle_lib/shock_calculator.py`**: Calculates pressure boundaries for different nozzle flow regimes and determines the location of a normal shock inside the nozzle if conditions dictate. Based on methods described by JoshTheEngineer.

## How to Run

1.  **Prerequisites:** Ensure you have Python 3 installed along with the required libraries (see Dependencies).

2.  Save the script and the `nozzle_lib` directory in the same location.

3.  Open a terminal or command prompt in that directory.

4.  **Configure Inputs:** Open `main_nozzle_analysis.py` and modify the variables in the `INPUTS` section:
    * Set physical constants (`M_MOLAR`, `GAMMA`).
    * Define gas state conditions (`P0`, `P_AMBIENT`).
    * Set fixed geometry (`R_CHAMBER_M`, `THETA_CONV_DEG`, `THETA_DIV_DEG`, `BELL_LENGTH_PERCENT`).
    * Define the ranges for the parameter sweep (`T0_START`, `T0_END`, `T0_STEPS`, `R_THROAT_M_START`, `R_THROAT_M_END`, `R_THROAT_M_STEPS`).
    * Adjust output controls (`PLOT_BELL_EACH_ITERATION`, `SAVE_CSV_EACH_ITERATION`, `PLOT_TRENDS_AT_END`).
5.  **Run:** Execute the main script from your terminal in the project's root directory:
    ```bash
    python main_nozzle_analysis.py
    ```
6.  The script will prompt you to select a mode:
    * Enter `1` for a single calculation using the default inputs defined globally.
    * Enter `2` for the parameter sweep defined in the `run_sweep_calculation` function.


## Outputs

* **Console:** Detailed logs of calculations, status updates, warnings, and final performance results. Progress and summary results for each iteration of the sweep will be printed to the console.
* **Plots:** Interactive plot windows displayed via Matplotlib. PNG files are **not** automatically saved in the current version, but the plotting code exists (commented out or inside modules) and could be re-enabled if needed.
* **CSV Files:**
    * `bell_nozzle_contour_ar<...>.csv` (Mode 1): Contains the [X, Y] coordinates of the calculated bell nozzle contour.
    * `sweep_results.csv` (Mode 2): Contains all calculated parameters for each iteration of the sweep.
    * `output/bell_contour_T<...>_Rt<...>.csv` (Mode 2, if `SAVE_CSV_EACH_ITERATION = True`): Individual bell contour files for valid iterations during the sweep.


## Dependencies

* **Python 3.x**
* **NumPy:** For numerical operations (`pip install numpy`)
* **SciPy:** For optimization (fsolve) (`pip install scipy`)
* **Matplotlib:** For plotting (`pip install matplotlib`)
* **(Optional) Pandas:** For saving sweep results to CSV easily (`pip install pandas`)

## Sources & Acknowledgements

* Rocket performance calculation from textbook: Rocket Propulsion Elements - George P Sutton, Oscar Biblarz
* The normal shock calculation logic (`shock_calculator.py`) is adapted from MATLAB code provided by JoshTheEngineer ([www.JoshTheEngineer.com](http://www.joshtheengineer.com/)).
* The Rao bell nozzle implementation (`bell_nozzle_module.py`) is based on the technical note "The thrust optimised parabolic nozzle" from aspirespace.org.uk ([Link to PDF](http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf)).

## License

MIT License
