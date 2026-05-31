import argparse
import json
import os
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
from utils.experiment_runner import (
    assign_speed_tiers,
    prepare_client_data,
    save_accuracy_json,
)
from utils.plot_accuracy import generate_all_plots

np.random.seed(42)
torch.manual_seed(42)


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
    training_data_clients = prepare_client_data(
        training_data, num_clients, dataset_info["num_classes"], is_non_iid
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

    labels = [label for label, _ in timeout_runs]
    distribution_name = "non_iid" if is_non_iid else "iid"
    save_accuracy_json(
        accuracy_history, labels, dataset_info["output_dir"],
        distribution_name, output_prefix
    )

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
