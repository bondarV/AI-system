#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== 1. Замір ДО оптимізації (before: qwen3:1.7b) ==="
python3 bench.py --base-url http://localhost:11434/v1 --model qwen3:1.7b --reasoning-effort none --prompts prompts_10.txt --label before --output ../bench_results.csv

echo "=== 2. Замір ПІСЛЯ оптимізації (after: qwen3:0.6b) ==="
python3 bench.py --base-url http://localhost:11434/v1 --model qwen3:0.6b --reasoning-effort none --prompts prompts_10.txt --label after --output ../bench_results.csv

echo "=== 3. Перевірка 8 тестових сценаріїв (Розділ 1.7) ==="
python3 test_scenarios.py --base-url http://localhost:11434/v1 --model qwen3:1.7b --output scenarios_results.md

echo "Готово! Результати збережено у ../bench_results.csv та scenarios_results.md"
