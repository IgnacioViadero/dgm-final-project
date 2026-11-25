#!/usr/bin/env bash
# Cadena de la GPU local (Qwen3-1.7B). Se lanza en paralelo con run_bench_endpoint.sh:
# usan recursos distintos (GPU de la maquina vs servicio central). Entre si, en serie:
# una sola generacion a la vez en la GPU compartida.
#
# Orden: primero el benchmark de los 90 y despues el smoke de 8. Es al contrario de lo que
# parece razonable, y la razon es el reloj: con la GPU compartida (20-11-2026 habia 64 GB
# ocupados por otro proceso) una corrida con thinking=1 tardaba 8.6 min por problema, asi que
# el smoke de 16 corridas se come 2.3 h de GPU para rellenar una casilla de una tabla de humo
# cuyo "funciona la cadena" ya esta contestado. El benchmark de los 90 es el que mide, y con
# el 1.7B es la unica configuracion que de verdad distingue arquitecturas (el 27B acierta
# 90/90 con las dos). Primero lo que mide.
set -u
cd /home/202212124/dgm/dgm-final-project
export PYTHONUNBUFFERED=1
export ARCA_LLM_BACKEND=hf
L=reports/logs

step () {  # nombre comando...
  local name="$1"; shift
  echo "### $name  inicio $(date +%F' '%H:%M:%S)"
  "$@" > "$L/$name.log" 2>&1
  local code=$?
  if [ $code -ne 0 ]; then
    echo "### $name  FALLO (exit=$code), reintento unico $(date +%H:%M:%S)"
    "$@" >> "$L/$name.log" 2>&1   # el benchmark reanuda: se salta lo ya hecho
    code=$?
    [ $code -ne 0 ] && echo "### $name  FALLO DEFINITIVO (exit=$code): se sigue con el resto"
  fi
  echo "### $name  fin exit=$code $(date +%F' '%H:%M:%S)"
}

# 1. Lo que mide: los 90 problemas con el 1.7B sin thinking.
ARCA_ENABLE_THINKING=0 \
  step numeric_hf_t0 uv run python -u -m eval.numeric_bench --out reports/numeric_hf_t0.json

# 2. La casilla que falta de la tabla del smoke (1.7B con thinking).
ARCA_ENABLE_THINKING=1 \
  step smoke_hf_t1 uv run python -u -m eval.smoke_agents --out reports/smoke_hf_t1.json

echo "### CADENA LOCAL TERMINADA $(date +%F' '%H:%M:%S)"
