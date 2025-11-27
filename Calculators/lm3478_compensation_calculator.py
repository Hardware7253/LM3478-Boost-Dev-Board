import math
import numpy as np
import matplotlib.pyplot as plt
import sys

# https://www.ti.com/lit/gpn/LM3478
# https://www.ti.com/lit/pdf/snva067

# User parameters 50V
# v_in = 5
# v_out = 50 
# i_out = 30e-3 # Max output current (mA)
# f = 460e3 # Switching frequency (Hz)

# l = 56e-6 # Inductor inductance (H)

# r_sen = 140e-3 # Current sensing resistor (Ω)

# c_out = 22e-6 # Output cap (F)
# c_esr = 50e-3 # Output cap equivalent series resistance (Ω)

# User parameters 75V
v_in = 5
v_out = 75 
i_out = 50e-3 # Max output current (mA)
f = 460e3 # Switching frequency (Hz)

l = 56e-6 # Inductor inductance (H)

r_sen = 75e-3 # Current sensing resistor (Ω)

c_out = 22e-6 # Output cap (F)
c_esr = 50e-3 # Output cap equivalent series resistance (Ω)

# User parameters 8V
# v_in = 5
# v_out = 8 
# i_out = 405e-3 # Max output current (mA)
# f = 140e3 # Switching frequency (Hz)

# l = 56e-6 # Inductor inductance (H)

# r_sen = 138e-3 # Current sensing resistor (Ω)

# c_out = 22e-6 # Output cap (F)
# c_esr = 50e-3 # Output cap equivalent series resistance (Ω)


r_load = v_out / i_out # Load output imedance (Ω)

# Device parameters
r_out = 50e3 # Error amplifier output impedance (Ω)
gm = 800e-6 # Error amplifier transconductance
v_fb = 1.26

# Calculation constants
PLOT_RES = 100 # Spacing between transfer function samples, 5 is high res, 100 is medium-low res (faster)

PM_TARGET = 55
PM_WEIGHT = 2
SLOPE_TARGET = -20
SLOPE_WEIGHT = 40

WIDE_COMP_CS = [1e-12, 100e-12, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6] # Wide sweep of compensations caps to start 
COMP_RS = [100, 1e3, 5e3, 10e3, 100e3] # Compensation resistors to test

# Multipliers for testing compensation combinations similiar to the best combination from the first sweep
SECOND_C_SWEEP_MIN = 0.1
SECOND_C_SWEEP_MAX = 10


# Search an array for the index of target
def naive_search(array, target, stop_tol):
    for i in range(len(array)):
        val = array[i]
        if (math.isclose(val, target, abs_tol=stop_tol)):
            return i

# Returns the approximate dB/dec of the transfer function at the crossover frequency
# Sample range is how many indices of the ws / mags arrays to check the slope between
def estimate_crossover_slope(crossover_index, ws, mags, sample_range):
    index_1 = crossover_index - sample_range // 2
    index_2 = crossover_index + sample_range // 2

    y1 = mags[index_1]
    y2 = mags[index_2]
    x1 = np.log10(ws[index_1])
    x2 = np.log10(ws[index_2])

    gradient = (y2 - y1) / (x2 - x1)
    return gradient

def calculate_parameters(c_comp, r_comp):
    # Calculate poles and zeros
    zero_1 = 1 / (c_out * c_esr)
    zero_2 = (r_load * (v_in / v_out) ** 2) / l
    zero_3 = 1 / (r_comp * c_comp)

    pole_1 = 2 / (c_out * r_load)
    pole_2 = 1 / (c_comp * r_out)
    complex_pole = 2 * math.pi * f

    # Gains
    duty = 1 - ((v_in) / (v_out))
    duty_p = 1 - duty

    a_cm = (duty_p * r_load) / (2 * r_sen)
    q = 1 / (math.pi * (duty_p * 2.1912 + 0.5 - duty))
    a_ea = gm * r_out
    a_fb = v_fb / v_out
    a_dc = a_cm * a_ea * a_fb

    # Transfer function
    H = lambda w, j=1j: a_dc * ((1 + j*w / zero_1) * (1 - j*w / zero_2) * (1 + j*w / zero_3) ) / ( (1 + j*w /  pole_1) * (1 + j*w / pole_2) * (1 + j*w / (q * complex_pole / 2) + ((j*w) ** 2) / ((complex_pole / 2) ** 2)))

    ws = np.arange(1, 1e7, PLOT_RES)
    mags = 20 * np.log10(np.abs(H(ws)))
    phases = np.angle(H(ws), deg=True)

    # Find the crossover frequency, phase margin, gain margin, and crossover slope
    crossover_index = naive_search(mags, 0, 0.1)
    if crossover_index is None:
        return (None, None, None, None, ws, mags, phases)

    crossover_frequency = ws[crossover_index] / (2 * math.pi)
    phase_margin = phases[crossover_index] + 180
    neg_180_deg_index = naive_search(phases, -180, 0.1)
    crossover_slope = estimate_crossover_slope(crossover_index, ws, mags, 20)

    gain_margin = None
    if neg_180_deg_index is not None:
        gain_margin = 0 - mags[neg_180_deg_index]

    return (crossover_frequency, phase_margin, crossover_slope, gain_margin, ws, mags, phases)

def plot(ws, mags, phases):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.semilogx(ws, mags)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, which='both')

    ax2.semilogx(ws, phases)
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both')

def calculate_and_plot(c_comp, r_comp):
    (crossover_frequency, phase_margin, crossover_slope, gain_margin, ws, mags, phases) = calculate_parameters(c_comp, r_comp)
    plot(ws, mags, phases)

    if crossover_frequency is not None:
        print(f"Crossover frequency: {crossover_frequency:.2f}Hz")
        print(f"Phase margin: {phase_margin:.2f} Degrees")
        print(f"Crossover slope: {crossover_slope:.2f} dB/decade")

        if gain_margin is not None:
            print(f"Gain margin: {gain_margin:.2f} dB")

# Calulcate a score for the performance of the compensation network
# A lower score is better
def calculate_performance_score(pm, slope):
    phase_score = abs(pm - PM_TARGET)
    slope_score = abs(slope - SLOPE_TARGET)
    score = phase_score * PM_WEIGHT  + slope_score * SLOPE_WEIGHT
    return score

# Scans all possible combinations of given compensation cs and rs
# Returns an array of tuples containing c, r, and performance parameters sorted by a performance score
def scan_compensation_combinations(comp_cs, comp_rs, verbose=False):
    if verbose:
        print(f"Searching {len(comp_cs) * len(comp_rs)} RC compensation combinations")

    scores_array = []

    for c in comp_cs:
        for r in comp_rs:

            (crossover_frequency, phase_margin, crossover_slope, gain_margin, ws, mags, phases) = calculate_parameters(c, r)
            if crossover_frequency is None:
                continue

            score = calculate_performance_score(phase_margin, crossover_slope)
            scores_array.append((score, phase_margin, crossover_slope, r, c))

    if verbose:
        print("Done")
    return sorted(scores_array)

# Prints the first (elements) scores
def print_scores(scores_array, elements):
    print(f"Best {elements} scores:")
    for (score, pm, slope, r, c) in scores_array[:elements]:
        print(f"Compensation combination C: {c}, R: {r} (pm: {pm}, slope: {slope})")

# Prints the best 5 combination options 
def find_good_compensation(verbose=False, plot=True):
    scores = scan_compensation_combinations(WIDE_COMP_CS, COMP_RS, verbose) 
    (_, _, _, _, best_c) = scores[0]

    new_comp_cs = np.arange(best_c * SECOND_C_SWEEP_MIN, best_c * SECOND_C_SWEEP_MAX, best_c * SECOND_C_SWEEP_MIN) # Hone in later
    scores = scan_compensation_combinations(new_comp_cs, COMP_RS, verbose) 
    (_, _, _, best_r, best_c) = scores[0]

    print("")
    print_scores(scores, 5)

    print("")
    print("Plot for best combination:")
    calculate_and_plot(best_c, best_r)



if len(sys.argv) == 1: # Provide no arguments for automatic compensation search and plot
    print("Crunching numbers very innefficiently")
    print("One moment please...")
    find_good_compensation(False)

    print("")
    print("Many factors other than the compensation cap and resistor effect the frequency response;")
    print("consider chaging output cap or switching frequency if the performance is undesirable.")
else: # If an argument is provided a manual value can be plotted
    calculate_and_plot(82e-12, 1e3) # 50V
    # calculate_and_plot(180e-12, 10e3) # 75V
    # calculate_and_plot(470e-9, 1e3) # 8V

plt.show()




