#!/usr/bin/env bash
# Una configuracion del banco de los 600 (90 de test + 510 de train) con el 1.7B local.
#
#   eval/run_bench_600.sh specialists
#   eval/run_bench_600.sh monolithic
#
# Se lanzan las dos a la vez, cada una en su proceso: el modelo son 3.45 GB en la slice MIG
# de 16 GB (medido el 20-11-2026), asi que caben dos copias y sobran 9 GB. Comparten los SM
# de la slice, de modo que en paralelo cada una va mas despacio que sola, pero el reloj de
# pared total baja y, sobre todo, un fallo en una no se lleva la otra por delante.
#
# Cada configuracion escribe su JSON, su sidecar y su log. El sidecar hace la corrida
# reanudable: si esto se muere en el problema 400, al relanzar el mismo comando se salta los
# 400 ya hechos. Si una configuracion falla por algo ajeno al codigo (la GPU ocupada, un OOM
# de otro proceso), se reintenta una sola vez y se reanuda desde donde estaba.
set -u
cd "$(dirname "$0")/.."

ARCH="${1:?uso: run_bench_600.sh specialists|monolithic}"
TAG="numeric_hf_t0_600_${ARCH}"
LOGS=reports/logs
mkdir -p "$LOGS" reports

export PYTHONUNBUFFERED=1
export ARCA_LLM_BACKEND=hf
export ARCA_ENABLE_THINKING=0

run () {
  uv run python -u -m eval.numeric_bench \
    --dataset data/rlm_data_test.jsonl \
    --dataset data/rlm_data_train.jsonl \
    --architectures "$ARCH" \
    --out "reports/${TAG}.json"
}

echo "### $TAG inicio $(date +%F' '%H:%M:%S)"
run >> "$LOGS/${TAG}.log" 2>&1
code=$?
if [ $code -ne 0 ]; then
  echo "### $TAG FALLO (exit=$code), reintento unico $(date +%F' '%H:%M:%S)"
  echo "### reintento $(date +%F' '%H:%M:%S)" >> "$LOGS/${TAG}.log"
  run >> "$LOGS/${TAG}.log" 2>&1   # reanuda: se salta lo ya hecho
  code=$?
  [ $code -ne 0 ] && echo "### $TAG FALLO DEFINITIVO (exit=$code)"
fi
echo "### $TAG fin exit=$code $(date +%F' '%H:%M:%S)"
exit $code
