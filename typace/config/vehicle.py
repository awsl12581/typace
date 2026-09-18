"""Vehicle resource and control thresholds."""

# Fractions of catalog battery capacity. Separate thresholds prevent rapid
# mode oscillation near brownout while preserving deterministic recovery.
POWER_SAFE_ENTER_FRACTION = 0.05
POWER_SAFE_RECOVER_FRACTION = 0.15

# Vehicle resources are authoritative; this tolerance only detects callers
# passing a stale duplicate total mass into the translational propagator.
MASS_CONSISTENCY_TOLERANCE_KG = 1.0e-6

# RK4 normalizes every attitude step; this bound allows floating-point rounding
# while treating externally corrupted, non-unit attitude state as a fault.
QUATERNION_NORM_TOLERANCE = 1.0e-9
