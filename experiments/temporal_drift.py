import argparse
import os
import sys

_BASE = os.path.dirname(__file__)
_SRC = os.path.join(_BASE, "..", "src")
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "synchronous"))
sys.path.insert(0, os.path.join(_SRC, "asynchronous"))

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


def _run_drift_sync(
    training_data,
    testing_data,
    dataset_info,
    schedule,
    phase_indices,
    num_clients,
    epochs,
    batch_size,
    percentile,
    eval_every,
    output_dir,
    output_prefix,
):
    from synchronous.client import Client
    from synchronous.constants import (
        MAX_CONNECTION_TIME,
        MIN_CONNECTION_TIME,
        SPEED_TIER_SEED,
        SPEED_TIERS,
    )
    from synchronous.monte_carlo import get_percentiles_timeout
    from synchronous.server import Server
    from utils.experiment_runner import assign_speed_tiers

    round_timeouts = get_percentiles_timeout(
        [percentile], MIN_CONNECTION_TIME, MAX_CONNECTION_TIME, SPEED_TIERS
    )
    # TODO: Porque ele pega o 0?
    round_timeout = round_timeouts[0]
    speed_assignments = assign_speed_tiers(num_clients, SPEED_TIERS, SPEED_TIER_SEED)

    # Assign phase 0 data
    x_full, y_full = training_data
    clients = []
    for i in range(num_clients):
        indices, _ = phase_indices[0][i]
        dataset = (x_full[indices], y_full[indices])
        clients.append(
            Client(
                dataset,
                i + 1,
                (speed_assignments[i][1], speed_assignments[i][2]),
                speed_assignments[i][0],
            )
        )

    group_masks = schedule.get_group_masks()
    server = Server(
        clients,
        num_clients,
        10**9,
        round_timeout,
        epochs,
        batch_size,
        testing_data,
        dataset_info["model"],
        evaluation_frequency=eval_every,
        group_masks=group_masks,
    )
    server.setup_clients()

    for phase in range(schedule.num_phases):
        before_len = len(server.accuracy_history)

        if phase == 0:
            server.start_training(stop_time=schedule.T_drift)
        else:
            for i, client in enumerate(server.clients):
                indices, _ = phase_indices[phase][i]
                client.dataset = (x_full[indices], y_full[indices])
                client.reset_optimizer()
            if hasattr(server, "reset_version"):
                server.reset_version()
            server.start_training(stop_time=schedule.T_drift)

        time_offset = phase * schedule.T_drift
        for entry in server.accuracy_history[before_len:]:
            entry["phase"] = phase + 1
            entry["time"] += time_offset

    key = f"T_drift_{int(schedule.T_drift)}_p{percentile}_sync"
    _save_drift_json(server.accuracy_history, key, output_dir,
                     "iid", output_prefix)


def _run_drift_async(
    training_data,
    testing_data,
    dataset_info,
    schedule,
    phase_indices,
    num_clients,
    epochs,
    batch_size,
    base_alpha,
    decay_of_base_alpha,
    tardiness_sensivity,
    eval_every,
    output_dir,
    output_prefix,
):
    from asynchronous.client import Client
    from asynchronous.constants import SPEED_TIER_SEED, SPEED_TIERS
    from asynchronous.server import Server
    from utils.experiment_runner import assign_speed_tiers

    speed_assignments = assign_speed_tiers(num_clients, SPEED_TIERS, SPEED_TIER_SEED)

    x_full, y_full = training_data
    clients = []
    for i in range(num_clients):
        indices, _ = phase_indices[0][i]
        dataset = (x_full[indices], y_full[indices])
        clients.append(
            Client(
                dataset,
                i + 1,
                (speed_assignments[i][1], speed_assignments[i][2]),
                speed_assignments[i][0],
            )
        )

    group_masks = schedule.get_group_masks()
    server = Server(
        clients,
        num_clients,
        0,
        schedule.total_time,
        epochs,
        batch_size,
        testing_data,
        dataset_info["model"],
        base_alpha,
        decay_of_base_alpha,
        tardiness_sensivity,
        evaluation_frequency=eval_every,
        group_masks=group_masks,
    )
    server.setup_clients()

    for phase in range(schedule.num_phases):
        before_len = len(server.accuracy_history)

        if phase == 0:
            server.start_training(stop_time=schedule.T_drift)
        else:
            for i, client in enumerate(server.clients):
                indices, _ = phase_indices[phase][i]
                client.dataset = (x_full[indices], y_full[indices])
                client.reset_optimizer()
            server.reset_version()
            server.start_training(stop_time=schedule.T_drift)

        time_offset = phase * schedule.T_drift
        for entry in server.accuracy_history[before_len:]:
            entry["phase"] = phase + 1
            entry["time"] += time_offset

    key = f"T_drift_{int(schedule.T_drift)}_async"
    _save_drift_json(server.accuracy_history, key, output_dir,
                     "iid", output_prefix)


def _save_drift_json(accuracy_history, key, output_dir, distribution_name,
                     output_prefix):
    import json

    prefix_str = f"_{output_prefix}" if output_prefix else ""
    filename = (
        f"accuracy_data_{distribution_name}"
        f"{prefix_str}.json"
    )
    data = {key: accuracy_history}
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Dados salvos em {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Experimento de Temporal Drift — Sync vs Async"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["sync", "async", "both"],
    )
    parser.add_argument("--T-drift", type=str, default="200",
                        help="Duracao de cada fase em segundos virtuais. "
                             "Multiplos valores separados por virgula (ex: 200,500,1000)")
    parser.add_argument("--num-phases", type=str, default="15",
                        help="Numero total de fases. Se multiplos T-drift, "
                             "valores correspondentes separados por virgula (ex: 15,6,5)")
    parser.add_argument("--num-clients", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--percentile", type=int, default=75)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--base-alpha", type=float, default=0.8)
    parser.add_argument("--decay-of-base-alpha", type=float, default=0.999)
    parser.add_argument("--tardiness-sensivity", type=float, default=0.075)
    args = parser.parse_args()

    from utils.data_loader import get_dataset_info, load_dataset
    from utils.drift import PhaseSchedule, precompute_phase_indices

    GROUP_A = [0, 1, 2, 3, 4]
    GROUP_B = [5, 6, 7, 8, 9]

    T_drifts = [float(v.strip()) for v in args.T_drift.split(",")]
    num_phases_list = [int(v.strip()) for v in args.num_phases.split(",")]
    if len(num_phases_list) == 1 and len(T_drifts) > 1:
        num_phases_list = num_phases_list * len(T_drifts)
    if len(num_phases_list) != len(T_drifts):
        parser.error(
            f"--num-phases tem {len(num_phases_list)} valores, "
            f"mas --T-drift tem {len(T_drifts)}. Devem ter o mesmo numero."
        )

    dataset_info = get_dataset_info(args.dataset)
    training_data, testing_data = load_dataset(args.dataset)
    _, y_train = training_data
    output_dir = dataset_info["output_dir"]

    for T_drift, num_phases in zip(T_drifts, num_phases_list):
        schedule = PhaseSchedule.create(
            T_drift=T_drift,
            num_phases=num_phases,
            group_A=GROUP_A,
            group_B=GROUP_B,
        )
        print(f"\n{'='*60}")
        print(f"Schedule: {schedule.num_phases} fases de {schedule.T_drift}s")
        print(f"Tempo total: {schedule.total_time}s")
        print(f"Grupos: A={GROUP_A}, B={GROUP_B}")

        phase_indices = precompute_phase_indices(
            y_train, args.num_clients, schedule, is_non_iid=False
        )

        base_prefix = args.output_prefix
        t_label = f"T{int(T_drift)}"

        if args.mode in ("sync", "both"):
            print("\n=== Modo Sincrono ===")
            sync_prefix = f"{base_prefix}_{t_label}_sync" if base_prefix else f"{t_label}_sync"
            _run_drift_sync(
                training_data, testing_data, dataset_info, schedule,
                phase_indices, args.num_clients, args.epochs, args.batch_size,
                args.percentile, args.eval_every, output_dir, sync_prefix,
            )

        if args.mode in ("async", "both"):
            print("\n=== Modo Assincrono ===")
            async_prefix = f"{base_prefix}_{t_label}_async" if base_prefix else f"{t_label}_async"
            _run_drift_async(
                training_data, testing_data, dataset_info, schedule,
                phase_indices, args.num_clients, args.epochs, args.batch_size,
                args.base_alpha, args.decay_of_base_alpha,
                args.tardiness_sensivity, args.eval_every,
                output_dir, async_prefix,
            )


if __name__ == "__main__":
    main()
