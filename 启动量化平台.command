#!/bin/zsh
cd "$(dirname "$0")"
LOG_FILE="./quant_platform_app.log"

echo "正在启动量化股票分析平台..."
echo "如果窗口没有出现，请把 quant_platform_app.log 的内容发给 Codex。"
echo

{
  echo "==== $(date) ===="
  echo "启动方式：command"
  ./.venv/bin/python ./qt_app.py
  echo "exit_code=$?"
} >> "$LOG_FILE" 2>&1
