import json
import os
import random as _stdlib_random

from utils.data_split import split_iid_data, split_non_iid_data

'''
Dado as configurações de tiers e o número de clientes, 
1. obtém a quantidade de clientes que existe em cada tier
2. cria uma lista com a quantidade de cada cliente.
3. embaralha a lista
'''
def assign_speed_tiers(num_clients, speed_tiers, seed):
    counts = []
    cumulative = 0
    for i, (_, _, _, prop) in enumerate(speed_tiers):
        if i < len(speed_tiers) - 1:
            count = int(round(num_clients * prop))
            counts.append(count)
            cumulative += count
        else:
            counts.append(num_clients - cumulative)
    assignments = []
    for (name, lo, hi, _), count in zip(speed_tiers, counts):
        assignments.extend([(name, lo, hi)] * count)
    rng = _stdlib_random.Random(seed)
    rng.shuffle(assignments)
    return assignments

'''
Obtém os dados IID ou NON-IID a depender da entrada
'''
def prepare_client_data(training_data, num_clients, num_classes, is_non_iid):
    if is_non_iid:
        print("Usando dados nao IID")
        return split_non_iid_data(training_data, num_clients, num_classes)
    else:
        print("Usando dados IID")
        return split_iid_data(training_data, num_clients, num_classes)

''' 
Dado uma lista de listas de dicionarios accuracy_history
accuracy_history = [
  [  # execução com percentile 75
    {"loss": 2.30, "accuracy": 0.10, "time": 125.45},
    {"loss": 2.10, "accuracy": 0.25, "time": 250.90},
    ...
  ],
  [  # execução com percentile 50
    {"loss": 2.30, "accuracy": 0.10, "time": 110.20},
    ...
  ],
]
Copia os valores para um arquivo json
'''
def save_accuracy_json(accuracy_history, labels, output_dir, distribution_name,
                       output_prefix=""):
    data = {}
    for i, label in enumerate(labels):
        percentile = accuracy_history[i]
        entries = [point.copy() for point in percentile]
        data[label] = entries
    prefix_str = f"_{output_prefix}" if output_prefix else ""
    accuracy_data_name = f"accuracy_data_{distribution_name}{prefix_str}.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{accuracy_data_name}", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dados salvos em {output_dir}/{accuracy_data_name}")
