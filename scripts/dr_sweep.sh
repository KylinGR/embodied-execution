#!/usr/bin/env bash
# 域随机化扫描：每个扰动单独一档，最后加一个「最坏组合」。
# 用法：bash scripts/dr_sweep.sh [level ...]    不带参数 = 全跑（约 25 分钟）
set -uo pipefail

cd ~/projects/embodied-execution
PY=$HOME/envs/diablo/bin/python
BASE=examples/xingtian_balance_analytic.json
OUT=runs/dr
MODEL=runs/A-fix28-stage2b/checkpoints/rl_model_100000_steps.zip
mkdir -p "$OUT"

make_cfg() {   # $1=输出json  $2=randomization 的 JSON 字符串
  "$PY" - "$BASE" "$1" "$2" <<'EOF'
import json, sys
base, out, rand = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
d = json.load(open(base, encoding="utf-8"))
d["environment"]["randomization"] = rand
json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
EOF
}

run_one() {    # $1=名字 $2=randomization
  local name=$1 rand=$2
  local cfg="$OUT/cfg-$name.json"
  make_cfg "$cfg" "$rand"
  echo "=== $name  $rand ==="
  rm -rf "$OUT/$name"
  "$PY" -u -m diablo_training.xingtian_diag2 \
      --model "$MODEL" --resolved "$cfg" \
      --seeds 2002 2003 2004 --out "$OUT/$name" 2>&1 | tail -2
}

declare -A LEVELS=(
  [base]='null'
  [delay10]='{"delay_ms":[10,10]}'
  [delay20]='{"delay_ms":[20,20]}'
  [delay10to20]='{"delay_ms":[10,20]}'
  [friction08]='{"friction":[0.8,0.8]}'
  [friction12]='{"friction":[1.2,1.2]}'
  [torque09]='{"torque":[0.9,0.9]}'
  [mass105]='{"mass":[1.05,1.05]}'
  [worst]='{"delay_ms":[20,20],"friction":[0.8,0.8],"mass":[1.05,1.05],"torque":[0.9,0.9]}'
)

if [ $# -gt 0 ]; then
  NAMES=("$@")
else
  NAMES=(base delay10 delay20 delay10to20 friction08 friction12 torque09 mass105 worst)
fi

for name in "${NAMES[@]}"; do
  run_one "$name" "${LEVELS[$name]:?unknown level: $name}"
done

echo
echo "=== 汇总 ==="
"$PY" scripts/dr_summary.py "$OUT"/*/diag.json
