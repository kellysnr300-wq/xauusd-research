"""Structural checks on the dashboard files (no browser needed).

Usage:
    python -m src.test_dashboard
"""

import re
from pathlib import Path

HTML = Path("dashboard/index.html").read_text()
JS = Path("dashboard/app.js").read_text()


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


tabs = sorted(
    (m.start(), m.group(1))
    for m in re.finditer(r'<section id="(\w+)" class="tab', HTML)
)


def tab_of(element_id):
    position = HTML.index(f'id="{element_id}"')
    owner = None
    for start, name in tabs:
        if start <= position:
            owner = name
    return owner


ids = re.findall(r'\sid="([\w-]+)"', HTML)
check("every element id is unique", len(ids) == len(set(ids)))

used = set(re.findall(r'\$\("([\w-]+)"\)', JS))
missing = sorted(i for i in used if f'id="{i}"' not in HTML)
check(f"every id app.js uses exists in index.html {missing or ''}", not missing)

PNL = ["pnlStrategy", "runButton", "runStatus", "runTimeframe", "runFirst",
       "runLast", "runParams", "runHistory", "runNotes", "kpiNet", "kpiPF",
       "accuracyPanel", "accuracyTag", "accWin", "accPnl", "accAmb",
       "accNote", "equityChart", "equityEmpty", "ddValue", "wlValue",
       "yearlyBody"]
STRATEGIES = ["strategy-list", "strategy-editor", "newStrategy",
              "strategyName", "strategyCode", "saveStrategy", "checkStrategy",
              "checkResult"]

for element_id in PNL:
    check(f"#{element_id} is inside the P&L tab", tab_of(element_id) == "pnl")
for element_id in STRATEGIES:
    check(f"#{element_id} is inside the Strategies tab",
          tab_of(element_id) == "strategies")

check("the three tabs exist", [n for _, n in tabs] == ["overview", "strategies", "pnl"])
print("\nAll dashboard structure tests passed.")
