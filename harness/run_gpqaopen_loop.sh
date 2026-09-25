#!/bin/zsh
# E2 answer collection with auto-resume (network has dropped mid-run). Per-cell cache makes each pass resume.
cd "$(dirname "$0")"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (data collection only)}"
MODELS="openai/gpt-5.5,anthropic/claude-opus-4.8,google/gemini-3.1-pro-preview,openai/gpt-5.4,x-ai/grok-4.3,anthropic/claude-sonnet-4.6,openai/gpt-5.1,moonshotai/kimi-k2.7-code,qwen/qwen3.7-max,z-ai/glm-5.2,deepseek/deepseek-v4-pro,qwen/qwen3-coder-plus,openai/gpt-5-mini,deepseek/deepseek-v3.2,minimax/minimax-m3,mistralai/mistral-large-2512,nvidia/nemotron-3-ultra-550b-a55b,google/gemini-3.5-flash"
for pass in 1 2 3 4 5 6; do
  echo "=== ANSWER PASS $pass ==="
  python3 -u experiment.py matrix --datasets gpqa_open --n 130 --tag marketGPQAOPEN --cap 600 --workers 10 --models "$MODELS" 2>&1
  CELLS=$(python3 -c "import json;R=json.load(open('../runs/matrix_marketGPQAOPEN.json'));print(sum(len(v['models']) for v in R.values()))" 2>/dev/null)
  echo "=== after pass $pass: ${CELLS:-0} answer-cells ==="
  if [ "${CELLS:-0}" -ge 2250 ]; then echo "=== ANSWERS COMPLETE ($CELLS) ==="; break; fi
  sleep 3
done
echo "=== GPQAOPEN ANSWERS DONE ==="
