#!/usr/bin/env bash
# 提交门禁：范围核对（可选）+ ruff F 类静态检查 + smoke 冒烟子集
#
# 用法:
#   scripts/gate.sh              # 全部检查
#   scripts/gate.sh --no-scope   # 跳过范围核对
#
# 规则（AGENTS.md「提交门禁」节）：任何提交前必须运行本脚本并通过；
# 未通过禁止提交（除非提交信息含 [skip-gate] 且说明原因）。
set -euo pipefail
cd "$(dirname "$0")/.."

# ── 范围纪律核对：.scope 声明清单 vs 实际改动 ──────────────────────
# 会话/任务开始时把"本次计划改动的文件清单"写进 .scope（每行一个路径，
# # 开头为注释）。有 .scope 时，实际改动超出清单即失败——防"顺手改动"。
# 没有声明清单时此步自动跳过，不阻塞。
if [[ "${1:-}" != "--no-scope" && -f .scope ]]; then
  echo "== 范围核对 (.scope) =="
  declared_but_untouched=0
  while IFS= read -r f; do
    [[ -z "$f" || "$f" == \#* ]] && continue
    if ! git -c core.quotepath=off status --porcelain -- "$f" | grep -q .; then
      echo "  [warn] 声明但未改动: $f"
      declared_but_untouched=1
    fi
  done < .scope
  violations="$(
    git -c core.quotepath=off status --porcelain | awk '{ $1=""; sub(/^ +/, ""); print }' |
    while IFS= read -r f; do
      grep -qxF -- "$f" .scope || echo "  [FAIL] 未声明: $f"
    done
  )"
  if [[ -n "$violations" ]]; then
    echo "$violations"
    echo "范围核对失败：实际改动超出 .scope 声明清单（补充声明或撤销改动）。"
    exit 1
  fi
  [[ "$declared_but_untouched" == 1 ]] && echo "  （有声明未改的条目，仅提示）"
  echo "  范围核对 OK"
fi

echo "== ruff (F) =="
uv run --active ruff check src tests scripts

echo "== ui 字面量门禁（tokens 单一数值源） =="
uv run --active python scripts/lint_ui_literals.py

echo "== smoke tests =="
uv run --active python -m pytest -m smoke -x -q

echo "== GATE PASS =="
