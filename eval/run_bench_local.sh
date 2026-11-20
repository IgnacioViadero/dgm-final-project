#!/usr/bin/env bash
# Cadena de la GPU local (Qwen3-1.7B). Se lanza en paralelo con run_bench_endpoint.sh:
# usan recursos distintos (GPU de la maquina vs servicio central). Entre si, en serie:
# una sola generacion a la vez en la GPU compartida.
set -u
cd /home/202212124/dgm/dgm-final-project
export PYTHONUNBUFFERED=1
L=reports/logs

step () {  # nombre comando...
  local name="$1"; shift
  echo "### $name  inicio $(date +%F' '%H:%M:%S)"
  "$@" > "$L/$name.log" 2>&1
  local code=$?
  if [ $code -ne 0 ]; then
    echo "### $name  FALLO (exit=$code), reintento unico $(date +%H:%M:%S)"
    "$@" >> "$L/$name.log" 2>&1   # reanuda: se salta lo ya hecho
    code=$?
    [ $code -ne 0 ] && echo "### $name  FALLO DEFINITIVO (exit=$code): se sigue con el resto"
  fi
  echo "### $name  fin exit=$code $(date +%F' '%H:%M:%S)"
}

# 1. Cierra el smoke que quedo a medias cuando se corto la sesion.
ARCA_ENABLE_THINKING=1 ARCA_LLM_BACKEND=hf \
  step smoke_hf_t1 uv run python -u -m eval.smoke_agents --out reports/smoke_hf_t1.json

# 2. Benchmark de los 90, 1.7B sin thinking (la configuracion que el smoke dejo mejor).
ARCA_ENABLE_THINKING=0 ARCA_LLM_BACKEND=hf \
  step numeric_hf_t0 uv run python -u -m eval.numeric_bench --out reports/numeric_hf_t0.json

echo "### CADENA LOCAL TERMINADA $(date +%F' '%H:%M:%S)"
