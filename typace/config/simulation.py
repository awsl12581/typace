"""Simulation cadence, limits, and deterministic ordering configuration."""

PHYSICS_STEP_SECONDS = 1.0
# Bounds one render-driven world update to deterministic fixed physics work.
MAX_PHYSICS_SUBSTEPS_PER_WORLD_STEP = 16
EPOCH_NORMALIZATION_STEP_SECONDS = 30.0
MAX_PENDING_COMMANDS_PER_SATELLITE = 32
MAX_SATELLITES_WITH_PERFORMANCE_GUARANTEE = 64
UI_REFRESH_HZ = 30.0
UPDATE_P95_BUDGET_SECONDS = 0.020
# NumPy/SciPy propagation releases the GIL. Four workers give the six bundled
# satellites useful overlap without oversubscribing typical desktop CPUs.
MAX_COMPUTE_WORKERS = 4
EVENT_TIME_TOLERANCE_SECONDS = 1.0e-3
# One-hour operational screening window; long enough to react within typical
# low-orbit periods without turning the render-driven tick into mission planning.
CONJUNCTION_PREDICTION_HORIZON_SECONDS = 3_600.0
MAX_CATALOG_BYTES = 4 * 1024 * 1024
MAX_CATALOG_OBJECTS = 1_024
MAX_CATALOG_STRING_LENGTH = 2_048
TIME_WARPS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
