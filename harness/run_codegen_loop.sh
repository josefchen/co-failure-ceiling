#!/bin/zsh
# Auto-resume wrapper for the hard-code matrix. Background runs have been SIGKILLed mid-pass
# (likely OOM); the per-cell disk cache (keyed on max_tokens=24k) makes each pass resume cheaply.
# Loop until coverage is complete (>=510/720 cells) or max passes, reconstructing after each pass.
cd "$(dirname "$0")"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (data collection only)}"
MODELS="openai/gpt-5.5,anthropic/claude-opus-4.8,google/gemini-3.1-pro-preview,openai/gpt-5.4,x-ai/grok-4.3,anthropic/claude-sonnet-4.6,openai/gpt-5.1,moonshotai/kimi-k2.7-code,qwen/qwen3.7-max,z-ai/glm-5.2,deepseek/deepseek-v4-pro,qwen/qwen3-coder-plus,openai/gpt-5-mini,deepseek/deepseek-v3.2,minimax/minimax-m3,mistralai/mistral-large-2512,nvidia/nemotron-3-ultra-550b-a55b,google/gemini-3.5-flash"
for pass in 1 2 3 4 5 6 7 8; do
  echo "=== PASS $pass (resumes from cache) ==="
  python3 -u experiment.py matrix --datasets codegen --n 200 --tag marketCG --cap 470 --workers 8 --models "$MODELS" 2>&1
  # rebuild matrix from whatever the cache holds and count covered cells
  python3 -u reconstruct.py --tag marketCG --datasets codegen --n 200 2>&1 | tail -1
  CELLS=$(python3 -c "import json;R=json.load(open('../runs/matrix_marketCG.json'));print(sum(len(v['models']) for v in R.values()))" 2>/dev/null)
  echo "=== after pass $pass: ${CELLS}/1134 cells ==="
  if [ "${CELLS:-0}" -ge 1115 ]; then echo "=== COVERAGE COMPLETE ($CELLS/1134) ==="; break; fi
  sleep 3
done
echo "=== CODEGEN LOOP DONE ==="
