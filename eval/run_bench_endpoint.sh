#!/usr/bin/env bash
# Cadena del servicio central (Qwen3.8-27B-FP8). En serie entre si: es un servicio
# compartido y no queremos medir latencias contaminadas por nuestras propias peticiones.
set -u
cd /home/202212124/dgm/dgm-final-project
export PYTHONUNBUFFERED=1
export ARCA_LLM_BACKEND=endpoint
L=reports/logs

step () {
  local name="$1"; shift
  echo "### $name  inicio $(date +%F' '%H:%M:%S)"
  "$@" > "$L/$name.log" 2>&1
  local code=$?
  if [ $code -ne 0 ]; then
    echo "### $name  FALLO (exit=$code), reintento unico $(date +%H:%M:%S)"
    "$@" >> "$L/$name.log" 2>&1
    code=$?
    [ $code -ne 0 ] && echo "### $name  FALLO DEFINITIVO (exit=$code): se sigue con el resto"
  fi
  echo "### $name  fin exit=$code $(date +%F' '%H:%M:%S)"
}

# 1-2. Los dos smoke que faltaban del 27B.
ARCA_ENABLE_THINKING=0 step smoke_ep_t0 uv run python -u -m eval.smoke_agents --out reports/smoke_ep_t0.json
ARCA_ENABLE_THINKING=1 step smoke_ep_t1 uv run python -u -m eval.smoke_agents --out reports/smoke_ep_t1.json

# 3-4. Benchmark de los 90 con el 27B, sin y con thinking.
ARCA_ENABLE_THINKING=0 step numeric_ep_t0 uv run python -u -m eval.numeric_bench --out reports/numeric_ep_t0.json
ARCA_ENABLE_THINKING=1 step numeric_ep_t1 uv run python -u -m eval.numeric_bench --out reports/numeric_ep_t1.json

echo "### CADENA ENDPOINT TERMINADA $(date +%F' '%H:%M:%S)"
