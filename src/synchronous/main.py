import argparse
import json
import os
import random as _stdlib_random
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from client import Client
from constants import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    MAX_CONNECTION_TIME,
    MIN_CONNECTION_TIME,
    NUM_CLIENTS,
    NUM_UPDATES,
    PERCENTILE_LIST,
    SPEED_TIER_SEED,
    SPEED_TIERS,
)
from monte_carlo import get_percentiles_timeout
from server import Server
from utils.data_loader import get_dataset_info, load_dataset
from utils.data_split import split_iid_data, split_non_iid_data
from utils.plot_accuracy import generate_all_plots

np.random.seed(42)
torch.manual_seed(42)


def assign_speed_tiers(num_clients, speed_tier_definitions, seed):
    """Distribui os clientes nos tiers e embaralha por client_id."""
    clients_per_tier = []
    assigned_client_count = 0
    for tier_index, (_, _, _, tier_proportion) in enumerate(speed_tier_definitions):
        if tier_index < len(speed_tier_definitions) - 1:
            tier_client_count = int(round(num_clients * tier_proportion))
            clients_per_tier.append(tier_client_count)
            assigned_client_count += tier_client_count
        else:
            clients_per_tier.append(num_clients - assigned_client_count)

    speed_tier_assignments = []
    for (tier_name, min_train_time, max_train_time, _), tier_client_count in zip(
        speed_tier_definitions, clients_per_tier
    ):
        speed_tier_assignments.extend(
            [(tier_name, min_train_time, max_train_time)] * tier_client_count
        )

    rng = _stdlib_random.Random(seed)
    rng.shuffle(speed_tier_assignments)
    return speed_tier_assignments


def main(
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    is_non_iid,
    dataset="cifar10",
    output_prefix="",
    single_percentile=None,
    include_no_timeout=False,
    evaluation_frequency=1,
):
    accuracy_history = []
    selected_percentiles = [single_percentile] if single_percentile else PERCENTILE_LIST
    local_epochs = epochs

    dataset_info = get_dataset_info(dataset)
    training_data, testing_data = load_dataset(dataset)
    if is_non_iid:
        print("Usando dados nao IID")
        training_data_clients = split_non_iid_data(
            training_data, num_clients, dataset_info["num_classes"]
        )
    else:
        print("Usando dados IID")
        training_data_clients = split_iid_data(
            training_data, num_clients, dataset_info["num_classes"]
        )

    timeout_by_percentile = get_percentiles_timeout(
        selected_percentiles,
        MIN_CONNECTION_TIME,
        MAX_CONNECTION_TIME,
        SPEED_TIERS,
    )
    client_speed_tiers = assign_speed_tiers(num_clients, SPEED_TIERS, SPEED_TIER_SEED)

    print("Distribuicao de tiers de velocidade dos clientes:")
    for tier_name, _, _, _ in SPEED_TIERS:
        client_count = sum(1 for tier in client_speed_tiers if tier[0] == tier_name)
        print(f"  {tier_name}: {client_count} cliente(s)")

    timeout_runs = [
        (str(percentile), timeout)
        for percentile, timeout in zip(selected_percentiles, timeout_by_percentile)
    ]
    if include_no_timeout:
        max_train_time = max(tier[2] for tier in SPEED_TIERS)
        timeout_runs.append(("include_no_timeout", MAX_CONNECTION_TIME + max_train_time))

    for timeout_label, round_timeout in timeout_runs:
        print(f"\nTimeout definido para '{timeout_label}': {round_timeout:.2f}s")
        clients = [
            Client(
                training_data_clients[client_index],
                client_index + 1,
                (
                    client_speed_tiers[client_index][1],
                    client_speed_tiers[client_index][2],
                ),
                client_speed_tiers[client_index][0],
            )
            for client_index in range(num_clients)
        ]

        server = Server(
            clients,
            num_clients,
            num_rounds,
            round_timeout,
            local_epochs,
            batch_size,
            testing_data,
            dataset_info["model"],
            evaluation_frequency=evaluation_frequency,
        )

        server.setup_clients()
        local_history = server.start_training()
        accuracy_history.append(local_history)

    data = {}
    for run_index, (timeout_label, _) in enumerate(timeout_runs):
        data[timeout_label] = [
            {"loss": p[0], "accuracy": p[1], "time": p[2]}
            for p in accuracy_history[run_index]
        ]
    distribution_name = "non_iid" if is_non_iid else "iid"
    output_prefix_suffix = f"_{output_prefix}" if output_prefix else ""
    accuracy_data_name = f"accuracy_data_{distribution_name}{output_prefix_suffix}.json"

    output_dir = dataset_info["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{accuracy_data_name}", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dados salvos em {output_dir}/{accuracy_data_name}")

    if not output_prefix:
        generate_all_plots(
            output_dir,
            is_non_iid,
            alpha=0.1,
            mode="Sincrono",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--num-rounds", type=int, default=NUM_UPDATES)
    parser.add_argument("--epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--non-iid", action="store_true", help="Roda apenas nao-IID")
    parser.add_argument("--iid", action="store_true", help="Roda apenas IID")
    parser.add_argument(
        "--include-no-timeout",
        action="store_true",
        help="Adiciona cenario sem timeout (T=MAX_CONNECTION+maior tier)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "mnist", "fashion_mnist", "gtsrb"],
        help="Dataset a usar (default: cifar10)",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=None,
        help="Percentil unico (ex: 50). Padrao: todos de PERCENTILE_LIST",
    )
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    if args.non_iid:
        distributions = [True]
    elif args.iid:
        distributions = [False]
    else:
        distributions = [False, True]

    for is_non_iid in distributions:
        main(
            args.num_clients,
            args.num_rounds,
            args.epochs,
            args.batch_size,
            is_non_iid,
            dataset=args.dataset,
            output_prefix=args.output_prefix,
            single_percentile=args.percentile,
            include_no_timeout=args.include_no_timeout,
            evaluation_frequency=args.eval_every,
        )
