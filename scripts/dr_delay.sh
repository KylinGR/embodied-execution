#!/usr/bin/env bash
# 延迟 × 预测补偿 扫描：固定扰动为纯控制延迟，逐档加补偿量。
# 用法：bash scripts/dr_delay.sh            # 全跑（2×5=10 档，约 25 分钟）
#       bash scripts/dr_delay.sh 20         # 只跑延迟 20 ms 的那几档
set -uo pipefail

cd ~/projects/embodied-execution
PY=$HOME/envs/diablo/bin/python
BASE=examples/xingtian_balance_analytic.json
OUT=runs/dr
MODEL=runs/A-fix28-stage2b/checkpoints/rl_model_100000_steps.zip
mkdir -p "$OUT"

make_cfg() {   # $1=输出  $2=延迟ms  $3=补偿ms
  "$PY" - "$BASE" "$1" "$2" "$3" <<'EOF'
import json, sys
base, out, delay, comp = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
d = json.load(open(base, encoding="utf-8"))
d["environment"]["randomization"] = {"delay_ms": [delay, delay]}
d["environment"]["delay_comp_ms"] = comp
json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
EOF
}

run_one() {    # $1=延迟  $2=补偿
  local d=$1 c=$2
  local name="d${d}_c${c}"
  local cfg="$OUT/cfg-$name.json"
  make_cfg "$cfg" "$d" "$c"
  echo "=== $name (delay ${d} ms, comp ${c} ms) ==="
  rm -rf "$OUT/$name"
  "$PY" -u -m diablo_training.xingtian_diag2 \
      --model "$MODEL" --resolved "$cfg" \
      --seeds 2002 2003 2004 --out "$OUT/$name" 2>&1 | tail -2 || true
}

if [ $# -ge 1 ]; then
  DELAYS=("$@")
else
  DELAYS=(10 20)
fi

for d in "${DELAYS[@]}"; do
  for c in 0 10 15 20 30; do
    run_one "$d" "$c"
  done
done

echo
echo "=== summary ==="
"$PY" scripts/dr_summary.py "$OUT"/d*_c*/diag.json
