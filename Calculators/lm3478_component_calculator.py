import math
import numpy as np
import sys

# https://www.ti.com/lit/gpn/LM3478

# User parameters - 50V
v_in = 5
v_out = 50 
i_out = 30e-3 # Max output current (mA)

vripple = 0.1
l = 56e-6

f = 460e3 # Desired switching frequency (Hz)

# User parameters - 75V
# v_in = 5
# v_out = 75 
# i_out = 50e-3 # Max output current (mA)

# vripple = 0.1
# l = 56e-6

# f = 460e3 # Desired switching frequency (Hz)

# User parameters - 8V
# v_in = 5
# v_out = 8 
# i_out = 405e-3 # Max output current (mA)

# vripple = 0.1
# l = 56e-6

# f = 138e3 # Desired switching frequency (Hz)

# MOSFET parameters
t_lh = 3.9e-9 # Rise time
t_hl = 1.9e-9 # Fall time
r_ds_on = 180e-3
v_mos = i_out * r_ds_on # MOSFET forward voltage drop

vf = 0.85 # Diode forward voltage drop

# Device parameters
v_sl = 92e-3
v_sl_ratio = 0.49
v_sense = 156e-3

# R sense multipliers for finding an appropriate slope compensation / current sense combination
R_SEN_MIN = 0.01
R_SEN_MAX = 100

duty = 1 - ((v_in - v_mos) / (v_out + vf))
print(f"Duty cycle: {(duty * 100):.3f}%")

lmin = (duty * (1 - duty) * v_in) / (2 * i_out * f)
print(f"Recommended inductor size >{(lmin):.3e}H (pick a value ~10x this)")


d_il_max = duty * v_in / (2 * f * l)
av_il_max = i_out / (1 - duty)
i_pk_sw = av_il_max + d_il_max
print(f"Switch peak current: {i_pk_sw:.3e}A")

i_cin_rms = d_il_max / math.sqrt(3)
i_cout_rms = math.sqrt((1 - duty) * ((i_out ** 2) * (duty / (1 - duty) ** 2) + (d_il_max ** 2) / 3))
c_out = (i_out * duty) / (f * vripple)
print("")
print(f"Input capacitor rms current: {i_cin_rms:.3e}A")
print(f"Output capacitor rms current: {i_cout_rms:.3e}A")
print(f"Recommended cap >{c_out:.3e}F (pick a value ~10x this)")

p_cond = av_il_max ** 2 * r_ds_on * duty
p_sw = (i_pk_sw * v_out / 2) * f * (t_lh + t_hl)
print("")
print(f"Mosfet conduction loss: {p_cond:.3e}W")
print(f"Mosfet switching loss: {p_sw:.3e}W")

r_sen = (v_sense - (duty * v_sense * v_sl_ratio)) / i_pk_sw
r_sen_max = (2 * v_sl * f * l) / (v_out - (2 * v_in))
print("")
print(f"Current sense resistor: {r_sen:.3e}Ω")


# Calculate slope compesation resistor and the new peak switch current
def calc_slope_compensation(r_sen):
    if (r_sen > r_sen_max):

        r_sl = (((r_sen * (v_out - 2 * v_in)) / (2 * f * l)) - v_sl) / 40e-6
        d_v_sl = 40e-6 * r_sl

        new_i_pk_sw = (v_sense - (duty * (v_sl + d_v_sl))) / r_sen
    
        return (r_sl, new_i_pk_sw)
    else:
        return (None, None)

def get_diff(val_1, val_2):
    return abs(val_1 - val_2)

# Attempts to automatically find a good slope compensation resistor
# This may fail and need to be solved manually or might have no good solution
def find_good_slope_compensation(init_r_sen, target_i_pk, iters=100000):
    (r_sl, new_i_pk_sw) = calc_slope_compensation(init_r_sen)

    if r_sl is None:
        return (None, None, None)

    # Current best slope compensation option ordered: r_sl, i_pk, r_sen, i_pk - target_i_pk difference
    best = (0, 0, 0, float('inf'))

    r_sens = np.linspace(init_r_sen * R_SEN_MIN, init_r_sen * R_SEN_MAX, iters)
    for r_sen in r_sens:
        (r_sl, new_i_pk_sw) = calc_slope_compensation(r_sen)

        if new_i_pk_sw is not None:
            diff = get_diff(target_i_pk, new_i_pk_sw)
            if best[3] > diff:
                best = (r_sl, new_i_pk_sw, r_sen, diff)

    return best[:3]


if len(sys.argv) == 1: # Provide no arguments for automatic slope compensation calculation
    (r_sl, new_i_pk_sw, r_sen) = find_good_slope_compensation(r_sen, i_pk_sw)
else: # Otherwise use manual values
    r_sen = 140e-3 # For 50V
    # r_sen = 75e-3 # For 75V
    # r_sen = 150e-3 # For 8 

    (r_sl, new_i_pk_sw) = calc_slope_compensation(r_sen)

if r_sl is not None and duty > 0.5:
    print(f"Slope compensation: ")
    print(f"Recommended R_sl >{r_sl:.3e}Ω (round up to nearest sensible value)")
    print(f"New r_sen: {r_sen:.3e}Ω")
    print(f"New switch peak current: {new_i_pk_sw:.3e}A")

print()