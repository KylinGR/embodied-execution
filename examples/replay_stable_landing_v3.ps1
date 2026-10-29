param([string]$OutputName = ('stable-landing-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss')))
$ErrorActionPreference = 'Stop'
if ($OutputName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'OutputName must contain only letters, digits, dash and underscore' }
$taskRoot = '/mnt/d/Codex/PPO_traning/embodied-execution'
& wsl -d Ubuntu-22.04 -- /usr/bin/env `
  DIABLO_ASSET_DIR=/home/qin/.local/share/embodied-execution `
  PYTHONPATH=/mnt/d/Codex/PPO_traning/embodied-execution/examples `
  MUJOCO_GL=egl `
  /home/qin/.venvs/embodied-execution-gpu/bin/python `
  "$taskRoot/examples/qualify_landing.py" `
  --model "$taskRoot/runs/stable-landing-ppo-v3-20261022/policy.zip" `
  --heights .025 .035 .045 --jitter .005 --seed-start 12000 --video `
  --output "$taskRoot/runs/$OutputName"
if ($LASTEXITCODE -ne 0) { throw "Qualification failed with exit code $LASTEXITCODE" }
Write-Output "Output: D:/Codex/PPO_traning/embodied-execution/runs/$OutputName"
