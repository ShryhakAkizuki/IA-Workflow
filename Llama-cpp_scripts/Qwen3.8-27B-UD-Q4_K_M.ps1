$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"preserve_thinking": true, "reasoning_effort": "max"}'

llama-server `
  -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M `
  --no-mmproj `
  -ngl all `
  -fa on `
  -ctk q8_0 -ctv q8_0 `
  -c 80128 `
  -ub 1024 -b 1024 `
  --parallel 1 `
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 `
  --presence-penalty 0.0 --repeat-penalty 1.0 `
  --jinja `
  --port 9000