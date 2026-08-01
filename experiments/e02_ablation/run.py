import argparse
import os
import subprocess

from experiments.shared.registry import temporary_output_path


def run_simulation(cmd):
    cmd_str = " ".join(cmd)
    print(f"\n[INICIANDO] {cmd_str}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCESSO] {cmd_str}")
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Código {e.returncode}: {cmd_str}")


def run_ablation(argv=None):
    parser = argparse.ArgumentParser(description="Estudo de ablação - variação isolada de parâmetros")
    parser.add_argument("--num-updates", type=int, default=40, help="Número de atualizações (default: 40)")
    parser.add_argument("--percentile", type=int, default=50, help="Percentil único para o ablation (default: 50)")
    parser.add_argument("--num-clients", type=int, default=40, help="Número de clientes (default: 40)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        default=str(temporary_output_path("e02-ablation", "cifar-10")),
    )
    args = parser.parse_args(argv)

    print("== Estudo de Ablação (variação isolada) ==\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # Valores padrão (referência)
    DEFAULT_ALPHA = 0.8
    DEFAULT_DECAY = 0.999
    DEFAULT_TARDINESS = 0.075

    # Valores a variar (um por vez, os outros ficam no padrão)
    alpha_values = [0.3, 0.5, 0.8]
    decay_values = [0.999, 0.99, 0.95]
    tardiness_values = [0.0, 0.075, 0.5]
    distributions = [False, True]  # IID, Non-IID

    # Monta experimentos one-at-a-time
    experiments = []

    for a in alpha_values:
        for dist in distributions:
            experiments.append((a, DEFAULT_DECAY, DEFAULT_TARDINESS, dist))

    for d in decay_values:
        for dist in distributions:
            experiments.append((DEFAULT_ALPHA, d, DEFAULT_TARDINESS, dist))

    for t in tardiness_values:
        for dist in distributions:
            experiments.append((DEFAULT_ALPHA, DEFAULT_DECAY, t, dist))

    # Remove duplicatas (o ponto padrão aparece nos 3 grupos)
    seen = set()
    unique = []
    for exp in experiments:
        if exp not in seen:
            seen.add(exp)
            unique.append(exp)

    print(f"Experimentos: {len(unique)}")
    print(f"Config: updates={args.num_updates}, percentil=p{args.percentile}, clientes={args.num_clients}")
    print(f"Referência: alpha={DEFAULT_ALPHA}, decay={DEFAULT_DECAY}, tardiness={DEFAULT_TARDINESS}\n")

    commands = []
    for (alpha, decay, tardiness, is_non_iid) in unique:
        suffix = f"async_A{alpha}_B{decay}_C{tardiness}_D{args.num_clients}"
        cmd = [
            "python", "src/asynchronous/main.py",
            "--num-clients", str(args.num_clients),
            "--num-updates", str(args.num_updates),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--base-alpha", str(alpha),
            "--decay-of-base-alpha", str(decay),
            "--tardiness-sensivity", str(tardiness),
            "--percentile", str(args.percentile),
            "--output-prefix", suffix,
            "--output-dir", args.output_dir,
            "--experiment-id", "e02-ablation",
        ]
        if is_non_iid:
            cmd.append("--non-iid")
        commands.append(cmd)

    print(f"Iniciando {len(commands)} experimentos sequencialmente...\n")
    for i, cmd in enumerate(commands, 1):
        print(f"--- Experimento {i}/{len(commands)} ---")
        run_simulation(cmd)

    # Gerar gráficos automaticamente
    print("\n== Gerando gráficos ==")
    for dist in ["iid", "non_iid"]:
        for vary in ["base_alpha", "decay", "tardiness"]:
            subprocess.run([
                "python", "experiments/plot_ablation.py",
                "--distribution", dist,
                "--percentile", str(args.percentile),
                "--vary", vary,
                "--output-dir", args.output_dir,
            ])

    print("\n== Estudo de ablação finalizado! ==")


if __name__ == "__main__":
    run_ablation()
