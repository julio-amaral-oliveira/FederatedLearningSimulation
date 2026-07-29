import subprocess
import json
import os
import sys

# --- CONFIGURAÇÕES ---
SCRIPT_PATH = "smoke_drift.py" 
OUTPUT_BASE_DIR = "output/severity_calibration"


CORRUPTIONS = ['gaussian_noise', 'frosted_glass_blur', 'motion_blur', 'fog']
SEVERITIES = [1,2,3,4,5]
TARGET_MIN_ACC = 0.25
TARGET_MAX_ACC = 0.50

def main():
    results_summary = {}

    for corruption in CORRUPTIONS:
        print(f"\n{'-'*40}")
        print(f" Analisando corrupção: {corruption}")
        print(f"{'-'*40}")
        
        optimal_severity = None
        
        for severity in SEVERITIES:
            print(f"Testando severidade {severity}...")
            
            # Cria um diretório isolado para cada teste para não sobrescrever resultados
            output_dir = os.path.join(OUTPUT_BASE_DIR, corruption, f"sev_{severity}")
            
            # Monta o comando de terminal para chamar o seu script original
            command = [
                sys.executable, SCRIPT_PATH,
                "--corruption", corruption,
                "--severity", str(severity),
                "--output-dir", output_dir
            ]
            
            try:
                # Executa o script. stdout=DEVNULL esconde os prints originais para manter o terminal limpo.
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                # Lê o resultado gerado (usamos o baseline.json, pois o impacto inicial do drift é idêntico ao agent.json)
                result_file = os.path.join(output_dir, "baseline.json")
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Procura a acurácia exata no momento da injeção do drift
                onset_acc = None
                for record in data.get("corrupted_accuracy_history", []):
                    if record.get("stage") == "drift_onset":
                        onset_acc = record.get("accuracy")
                        break
                
                if onset_acc is None:
                    print("  [!] Aviso: Não foi possível encontrar a acurácia de 'drift_onset' no JSON.")
                    continue
                
                print(f"  -> Acurácia obtida: {onset_acc:.4f}")
                
                # Verifica se a acurácia caiu para o alvo desejado (entre 0.25 e 0.5)
                if TARGET_MIN_ACC <= onset_acc <= TARGET_MAX_ACC:
                    print(f"  [V] Sucesso! Menor severidade ideal encontrada: {severity}.")
                    optimal_severity = severity
                    
                    # Salva os dados no resumo
                    results_summary[corruption] = {
                        "optimal_severity": severity,
                        "accuracy": onset_acc,
                        "result_path": output_dir
                    }
                    
                    # Interrompe o loop de severidade e vai para o próximo tipo de corrupção
                    break 
                    
            except subprocess.CalledProcessError as e:
                print(f"  [X] Erro ao executar o script original:\n{e.stderr.decode()}")
                
        # Se testou de 1 a 5 e nenhuma atingiu o alvo
        if optimal_severity is None:
            print(f"  [!] Nenhuma severidade entre {SEVERITIES[0]} e {SEVERITIES[-1]} deixou a acurácia no alvo estipulado para {corruption}.")
            results_summary[corruption] = {
                "optimal_severity": None,
                "note": "Fora do alvo (0.25 - 0.50) para todas as severidades"
            }

    # --- SALVANDO O RELATÓRIO FINAL ---
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    summary_file = os.path.join(OUTPUT_BASE_DIR, "optimal_severities_summary.json")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'-'*40}")
    print(f"Busca concluída! Resumo consolidado salvo em:\n{summary_file}")

if __name__ == "__main__":
    main()