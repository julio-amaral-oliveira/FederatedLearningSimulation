# constants.py

MIN_CONNECTION_TIME = 0
MAX_CONNECTION_TIME = 1
NUM_CLIENTS = 40
NUM_UPDATES = 80
TIMEOUT = 8
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
PERCENTILE_LIST = [25, 50, 75]

# Heterogeneidade intrinseca de capacidade computacional dos clientes.
# Cada tier: (nome, min_train_time, max_train_time, proporcao_de_clientes)
SPEED_TIERS = [
    ("fast", 1, 5, 0.50),
    ("medium", 5, 12, 0.20),
    ("slow", 12, 25, 0.20),
    ("very_slow", 25, 40, 0.10),
]
SPEED_TIER_SEED = 42
SIMULATION_SEED = 42
