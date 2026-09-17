"""Physical constants shared by simulation modules.

Values use SI units. Astronomical constants follow the IAU 2015 nominal
values where available; Earth J2 follows EGM2008 and standard gravity follows
the 3rd CGPM conventional value.
"""

from astropy import units

STANDARD_GRAVITY_M_S2 = 9.80665
ASTRONOMICAL_UNIT_M = units.au.to(units.m)
SOLAR_CONSTANT_W_M2 = 1361.0

EARTH_GRAVITATIONAL_PARAMETER_M3_S2 = 3.986004418e14
EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0
EARTH_MEAN_RADIUS_M = 6_371_008.4
EARTH_J2 = 1.08262668e-3
EARTH_ROTATION_RATE_RAD_S = 7.2921150e-5

MOON_GRAVITATIONAL_PARAMETER_M3_S2 = 4.902800118e12
MOON_MEAN_RADIUS_M = 1_737_400.0

# Sutton-Graves Earth-air coefficient for SI inputs: W/m2, kg/m3, m, m/s.
SUTTON_GRAVES_EARTH_COEFFICIENT = 1.7415e-4
