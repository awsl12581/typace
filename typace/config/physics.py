"""Physical constants shared by simulation modules.

Values use SI units. Astronomical constants follow the IAU 2015 nominal
values where available; Earth J2 follows EGM2008 and standard gravity follows
the 3rd CGPM conventional value.
"""

STANDARD_GRAVITY_M_S2 = 9.80665
# IAU 2012 Resolution B2 exact astronomical unit definition.
ASTRONOMICAL_UNIT_M = 149_597_870_700.0
SOLAR_CONSTANT_W_M2 = 1361.0

EARTH_GRAVITATIONAL_PARAMETER_M3_S2 = 3.986004418e14
EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0
EARTH_MEAN_RADIUS_M = 6_371_008.4
EARTH_J2 = 1.08262668e-3
EARTH_ROTATION_RATE_RAD_S = 7.2921150e-5

# Piecewise exponential engineering atmosphere from Vallado's adaptation of
# the 1976 US Standard Atmosphere/CIRA model. The table covers 0-1000 km;
# base densities are derived continuously from sea level at runtime.
EARTH_ATMOSPHERE_SEA_LEVEL_DENSITY_KG_M3 = 1.225
EARTH_ATMOSPHERE_LAYER_ALTITUDES_M = (
    0.0,
    25_000.0,
    30_000.0,
    40_000.0,
    50_000.0,
    60_000.0,
    70_000.0,
    80_000.0,
    90_000.0,
    100_000.0,
    110_000.0,
    120_000.0,
    130_000.0,
    140_000.0,
    150_000.0,
    180_000.0,
    200_000.0,
    250_000.0,
    300_000.0,
    350_000.0,
    400_000.0,
    450_000.0,
    500_000.0,
    600_000.0,
    700_000.0,
    800_000.0,
    900_000.0,
    1_000_000.0,
)
EARTH_ATMOSPHERE_SCALE_HEIGHTS_M = (
    7_249.0,
    6_349.0,
    6_682.0,
    7_554.0,
    8_382.0,
    7_714.0,
    6_549.0,
    5_799.0,
    5_382.0,
    5_877.0,
    7_263.0,
    9_473.0,
    12_636.0,
    16_149.0,
    22_523.0,
    29_740.0,
    37_105.0,
    45_546.0,
    53_628.0,
    53_298.0,
    58_515.0,
    60_828.0,
    63_822.0,
    71_835.0,
    88_667.0,
    124_640.0,
    181_050.0,
    268_000.0,
)

MOON_GRAVITATIONAL_PARAMETER_M3_S2 = 4.902800118e12
MOON_MEAN_RADIUS_M = 1_737_400.0

# Sutton-Graves Earth-air coefficient for SI inputs: W/m2, kg/m3, m, m/s.
SUTTON_GRAVES_EARTH_COEFFICIENT = 1.7415e-4
