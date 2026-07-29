# Fix 4 — relatório de partição de `summary.json`

## Resultado

`summary.json` schema 1 agora contém apenas `groups`; não existe mais uma
métrica agregada global que misture cenários visuais e controles. Cada grupo
usa uma chave JSON estável, derivada do descritor explícito da execução, e
persiste:

- dimensões: população, corrupção, severidade, política de gatilho, quorum,
  rounds de retreino e horizonte de produção;
- sementes únicas ordenadas e `run_count`;
- métricas finitas de `agent` e `baseline` separadas.

O descritor vem do `DriftEpisodeConfig` efetivamente passado ao runner e da
política de monitor efetivamente escolhida pelo `run_matrix`; nenhum dado é
inferido da estrutura de diretórios. Uma execução normal de identidade é
`identity_control`/`detector`, enquanto o controle criado por
`--include-controls` é `oracle_control`/`oracle`.

## TDD: red

Antes da implementação, executei:

```bash
.venv/bin/python -m unittest \
  tests.test_drift_runner.TestDriftRunMatrix.test_summary_partitions_identity_control_from_visual_metrics \
  tests.test_drift_runner.TestDriftRunMatrix.test_summary_partitions_oracle_from_detector_triggered_runs \
  tests.test_drift_runner.TestDriftRunMatrix.test_summary_aggregates_equal_dimensions_across_sorted_unique_seeds -v
```

Os três testes falharam pela ausência de `groups`. O primeiro também expôs o
defeito: o resumo anterior calculava média `51.0` para os valores visual `2.0`
e controle de identidade `100.0`.

## TDD: green

Após a implementação mínima, o mesmo comando passou com 3 testes. Em seguida:

```bash
.venv/bin/python -m unittest tests.test_drift_runner -v
# Ran 11 tests ... OK

.venv/bin/python -m unittest discover -s tests -v
# Ran 108 tests ... OK
```

A suíte completa inclui `test_summary_write_failure_preserves_existing_destination`;
ela continua verde, confirmando que o resumo é publicado com
`atomic_write_json` e uma falha não destrói o destino anterior.

## Limites conhecidos

O formato schema 1 de `summary.json` foi intencionalmente remodelado para o
contrato de grupos. Consumidores que liam os antigos campos globais `agent` e
`baseline` devem passar a iterar `groups` e selecionar dimensões compatíveis.
