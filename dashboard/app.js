const $ = (id) => document.getElementById(id);

const TOKEN =
    document.querySelector('meta[name="api-token"]')?.content || "";

const navItems = document.querySelectorAll(".nav-item");
const tabs = document.querySelectorAll(".tab");
const selectors = document.querySelectorAll(".strategy-selector");

const TEMPLATE = `import pandas as pd

from src.strategies import Strategy


class MyStrategy(Strategy):
    """Describe the hypothesis here."""

    def __init__(self, length=20):
        self.length = length

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # data columns: timestamp, open, high, low, close, volume
        #
        # Return ONE row per candle. "signal" is the position you want
        # AFTER that candle closes: 1 = long, -1 = short, 0 = flat.
        # The engine trades it at the NEXT candle's open.
        #
        # Use only current and past candles. No shift(-1), and no
        # statistics of the whole dataset: the look-ahead check rejects those.
        #
        # Optional columns: stop_distance, target_distance (price units
        # from entry).
        average = data["close"].rolling(self.length).mean()

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals.loc[data["close"] > average, "signal"] = 1
        signals.loc[data["close"] < average, "signal"] = -1
        return signals
`;

let strategies = [];
let selectedId = null;
let editingId = null;
let editingStatus = "draft";
let pollTimer = null;
let currentDetail = null;


/* ---------- helpers ---------- */

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2600);
}

async function api(path, method = "GET", body) {
    const options = { method, headers: { "X-Token": TOKEN } };

    if (method === "POST") {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body ?? {});
    }

    const response = await fetch("/api" + path, options);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data.error || response.statusText);
    }

    return data;
}

function money(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return "—";
    }
    const sign = value > 0 ? "+" : value < 0 ? "−" : "";
    return sign + Math.abs(value).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    });
}

function signClass(value) {
    return value > 0 ? "pos" : value < 0 ? "neg" : "";
}

function parseJsonField(text) {
    const trimmed = text.trim();
    if (!trimmed) return {};
    const value = JSON.parse(trimmed);
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new Error("Parameters must be a JSON object, e.g. {\"length\": 20}");
    }
    return value;
}


/* ---------- tabs ---------- */

function showTab(target) {
    navItems.forEach((item) => {
        item.classList.toggle("active", item.dataset.tab === target);
    });
    tabs.forEach((tab) => {
        tab.classList.toggle("active", tab.id === target);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });

    if (target === "pnl") drawEquity();
}

navItems.forEach((item) => {
    item.addEventListener("click", () => showTab(item.dataset.tab));
});


/* ---------- strategies ---------- */

async function loadStrategies() {
    const data = await api("/strategies");
    strategies = data.strategies;
    renderStrategies();
    populateSelectors();
}

function renderStrategies() {
    const list = $("strategy-list");

    if (!strategies.length) {
        list.innerHTML =
            '<p class="empty-note">No strategies yet. Create one with "+ New strategy".</p>';
        return;
    }

    list.innerHTML = strategies.map((s) => `
        <div class="strategy-card">
            <div class="strategy-card-top">
                <span class="strategy-type">${escapeHtml(s.type.toUpperCase())}</span>
                <span class="tag ${s.status === "saved" ? "saved" : "draft"}">
                    ${s.status === "saved" ? "Saved" : "Draft"}
                </span>
            </div>
            <h3>${escapeHtml(s.name)}</h3>
            <p>${escapeHtml(s.description || "No description.")}</p>
            <div class="strategy-footer">
                <span>${s.experiments} experiment${s.experiments === 1 ? "" : "s"}</span>
                <span class="card-actions">
                    <button class="text-button" data-action="edit" data-id="${s.id}">Edit</button>
                    <button class="text-button" data-action="run" data-id="${s.id}">Run</button>
                </span>
            </div>
        </div>
    `).join("");
}

$("strategy-list").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;

    if (button.dataset.action === "edit") {
        openEditor(button.dataset.id);
    } else {
        selectStrategy(button.dataset.id);
        showTab("pnl");
    }
});

function populateSelectors() {
    if (!strategies.some((s) => s.id === selectedId)) {
        selectedId = strategies.length ? strategies[0].id : null;
    }

    const options = strategies.length
        ? strategies.map((s) =>
            `<option value="${s.id}">${escapeHtml(s.name)}</option>`).join("")
        : '<option value="">No strategies</option>';

    selectors.forEach((select) => {
        select.innerHTML = options;
        select.value = selectedId || "";
    });

    onStrategyChanged();
}

function selectStrategy(id) {
    selectedId = id;
    selectors.forEach((select) => { select.value = id; });
    onStrategyChanged();
}

selectors.forEach((select) => {
    select.addEventListener("change", (event) => {
        selectStrategy(event.target.value);
    });
});

function onStrategyChanged() {
    const strategy = strategies.find((s) => s.id === selectedId);
    $("runParams").value = strategy ? JSON.stringify(strategy.params || {}) : "";
    loadHistory(true);
}


/* ---------- editor ---------- */

async function openEditor(id) {
    const editor = $("strategy-editor");
    hideCheck();

    if (id) {
        try {
            const s = await api("/strategies/" + id);
            editingId = s.id;
            editingStatus = s.status;
            $("editorTitle").textContent = "Edit strategy";
            $("strategyName").value = s.name;
            $("strategyType").value = s.type;
            $("strategyDescription").value = s.description;
            $("strategyParams").value = JSON.stringify(s.params || {});
            $("strategyCode").value = s.code;
        } catch (error) {
            showToast(error.message);
            return;
        }
    } else {
        editingId = null;
        editingStatus = "draft";
        $("editorTitle").textContent = "Create strategy";
        $("strategyName").value = "";
        $("strategyType").value = "Other";
        $("strategyDescription").value = "";
        $("strategyParams").value = '{"length": 20}';
        $("strategyCode").value = TEMPLATE;
    }

    editor.classList.remove("hidden");
    editor.scrollIntoView({ behavior: "smooth", block: "start" });
}

$("newStrategy").addEventListener("click", () => openEditor(null));
$("closeEditor").addEventListener("click", () => {
    $("strategy-editor").classList.add("hidden");
});

function hideCheck() {
    $("checkResult").className = "check-result hidden";
}

function showCheck(ok, text) {
    const box = $("checkResult");
    box.className = "check-result " + (ok ? "ok" : "bad");
    box.textContent = text;
}

async function saveCurrent(status) {
    const name = $("strategyName").value.trim();

    if (!name) {
        showToast("Enter a strategy name first.");
        $("strategyName").focus();
        return null;
    }

    let params;
    try {
        params = parseJsonField($("strategyParams").value);
    } catch (error) {
        showToast("Parameters: " + error.message);
        return null;
    }

    try {
        const saved = await api("/strategies", "POST", {
            id: editingId,
            name,
            type: $("strategyType").value,
            description: $("strategyDescription").value,
            params,
            code: $("strategyCode").value,
            status,
        });
        editingId = saved.id;
        editingStatus = saved.status;
        await loadStrategies();
        return saved;
    } catch (error) {
        showCheck(false, error.message);
        return null;
    }
}

$("saveDraft").addEventListener("click", async () => {
    const saved = await saveCurrent("draft");
    if (saved) showToast(`Draft saved: ${saved.name}`);
});

$("saveStrategy").addEventListener("click", async () => {
    const saved = await saveCurrent("saved");
    if (saved) {
        showToast(`Strategy saved: ${saved.name}`);
        $("strategy-editor").classList.add("hidden");
    }
});

$("checkStrategy").addEventListener("click", async () => {
    const saved = await saveCurrent(editingStatus);
    if (!saved) return;

    showCheck(true, "Checking…");

    try {
        const result = await api(`/strategies/${saved.id}/check`, "POST", {});
        if (result.ok) {
            showCheck(true,
                "Check passed: output is valid and uses no future data " +
                `(${result.info.synthetic_trades} trades on synthetic data).`);
        } else {
            showCheck(false, result.problems.join("\n"));
        }
    } catch (error) {
        showCheck(false, error.message);
    }
});


/* ---------- running backtests ---------- */

function setRunStatus(text, kind = "") {
    const tag = $("runStatus");
    tag.textContent = text;
    tag.className = "tag " + kind;
}

$("runButton").addEventListener("click", async () => {
    if (!selectedId) {
        showToast("Create a strategy first.");
        return;
    }

    let params;
    try {
        params = parseJsonField($("runParams").value);
    } catch (error) {
        showToast("Parameters: " + error.message);
        return;
    }

    try {
        const { run_id } = await api("/runs", "POST", {
            strategy_id: selectedId,
            params,
            first_year: Number($("runFirst").value),
            last_year: Number($("runLast").value),
            spread: Number($("runSpread").value),
            slippage: Number($("runSlippage").value),
            quantity: Number($("runQty").value),
        });
        pollRun(run_id);
    } catch (error) {
        showToast(error.message);
    }
});

function pollRun(runId) {
    clearInterval(pollTimer);
    const started = Date.now();
    $("runButton").disabled = true;

    const tick = async () => {
        try {
            const detail = await api("/runs/" + runId);
            const stage = detail.status.stage;
            const seconds = Math.round((Date.now() - started) / 1000);

            if (stage === "done") {
                clearInterval(pollTimer);
                $("runButton").disabled = false;
                setRunStatus("Done", "ok");
                await loadStrategies();
                await loadHistory(false, runId);
                return;
            }

            if (stage === "failed") {
                clearInterval(pollTimer);
                $("runButton").disabled = false;
                setRunStatus("Failed", "bad");
                showToast(detail.status.message || "Backtest failed.");
                return;
            }

            setRunStatus(
                `${detail.status.message || stage} · ${seconds}s`, "busy");
        } catch (error) {
            clearInterval(pollTimer);
            $("runButton").disabled = false;
            setRunStatus("Error", "bad");
            showToast(error.message);
        }
    };

    tick();
    pollTimer = setInterval(tick, 1500);
}

async function loadHistory(autoShowLatest, preferRunId) {
    const select = $("runHistory");

    if (!selectedId) {
        select.innerHTML = "";
        renderRun(null);
        return;
    }

    try {
        const { runs } = await api("/runs?strategy=" + selectedId);
        const done = runs.filter((r) => r.stage === "done");

        select.innerHTML = done.length
            ? done.map((r) => {
                const when = (r.created || r.run_id.slice(0, 15))
                    .replace("T", " ").slice(0, 16);
                const pf = r.profit_factor === null || r.profit_factor === undefined
                    ? "—" : r.profit_factor.toFixed(2);
                return `<option value="${r.run_id}">${when} · ` +
                    `${r.first_year ?? "?"}–${r.last_year ?? "?"} · ` +
                    `PnL ${money(r.total_pnl, 0)} · PF ${pf}</option>`;
            }).join("")
            : '<option value="">No runs yet</option>';

        if (done.length && (autoShowLatest || preferRunId)) {
            const target = preferRunId || done[0].run_id;
            select.value = target;
            await showRun(target);
        } else if (!done.length) {
            renderRun(null);
            setRunStatus("Idle");
        }
    } catch (error) {
        showToast(error.message);
    }
}

$("runHistory").addEventListener("change", (event) => {
    if (event.target.value) showRun(event.target.value);
});

async function showRun(runId) {
    try {
        currentDetail = await api("/runs/" + runId);
        renderRun(currentDetail);
        if (currentDetail.status.stage === "done") setRunStatus("Done", "ok");
    } catch (error) {
        showToast(error.message);
    }
}


/* ---------- rendering results ---------- */

function renderRun(detail) {
    currentDetail = detail;
    const s = detail && detail.summary;

    if (!s) {
        ["kpiNet", "kpiPF", "kpiAvg", "kpiRR", "ddValue", "wlValue"]
            .forEach((id) => { $(id).textContent = "—"; $(id).className = ""; });
        $("kpiNetSub").textContent = "No backtest data";
        $("kpiPFSub").textContent = "Not calculated";
        $("kpiAvgSub").textContent = "Not calculated";
        $("kpiRRSub").textContent = "Not calculated";
        $("ddSub").textContent = "Maximum drawdown";
        $("wlSub").textContent = "Trade distribution";
        $("equityTag").textContent = "Awaiting data";
        $("yearlyBody").innerHTML =
            '<tr><td class="muted-cell">No data</td></tr>';
        $("equityChart").classList.add("hidden");
        $("equityEmpty").classList.remove("hidden");
        return;
    }

    const qty = s.config?.quantity ?? 1;

    $("kpiNet").textContent = money(s.total_pnl, 0);
    $("kpiNet").className = signClass(s.total_pnl);
    $("kpiNetSub").textContent =
        `${(s.trade_count ?? 0).toLocaleString()} trades · ${s.first_year ?? "?"}–${s.last_year ?? "?"}`;

    $("kpiPF").textContent =
        s.profit_factor == null ? "—" : s.profit_factor.toFixed(2);
    $("kpiPF").className = s.profit_factor >= 1 ? "pos" : "neg";
    $("kpiPFSub").textContent = s.profit_factor >= 1
        ? "Wins outweigh losses" : "Losses outweigh wins";

    $("kpiAvg").textContent = money(s.average_pnl, 3);
    $("kpiAvg").className = signClass(s.average_pnl);
    $("kpiAvgSub").textContent = `per trade (${qty} oz)`;

    $("kpiRR").textContent =
        s.payoff_ratio == null ? "—" : s.payoff_ratio.toFixed(2);
    $("kpiRRSub").textContent = "avg win ÷ avg loss";

    $("ddValue").textContent = money(-(s.max_drawdown ?? 0), 0);
    $("ddValue").className = "neg";
    $("ddSub").textContent =
        `Maximum drawdown · ${(s.max_drawdown_pct ?? 0).toFixed(1)}% of capital`;

    $("wlValue").textContent = `${(s.win_rate ?? 0).toFixed(1)}% wins`;
    $("wlSub").textContent =
        `${(s.wins ?? 0).toLocaleString()} wins / ${(s.losses ?? 0).toLocaleString()} losses · ` +
        `avg win ${money(s.avg_win, 2)} · avg loss ${money(s.avg_loss, 2)}`;

    const years = Object.entries(s.yearly_pnl || {});
    $("yearlyBody").innerHTML = years.length
        ? years.map(([year, value]) =>
            `<tr><td>${year}</td><td class="${signClass(value)}">` +
            `${money(value, 2)}</td></tr>`).join("")
        : '<tr><td class="muted-cell">No data</td></tr>';

    $("equityTag").textContent =
        `${s.params && Object.keys(s.params).length
            ? JSON.stringify(s.params) : "no params"}`;
    $("equityEmpty").classList.add("hidden");
    $("equityChart").classList.remove("hidden");
    drawEquity();
}

function drawEquity() {
    const canvas = $("equityChart");
    const detail = currentDetail;

    if (!detail || !detail.equity || canvas.classList.contains("hidden")) {
        return;
    }

    const { equity, start, first, last } = detail.equity;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (!width) return;

    canvas.width = width * ratio;
    canvas.height = height * ratio;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const values = [start, ...equity];
    let low = Math.min(...values);
    let high = Math.max(...values);
    if (high === low) { high += 1; low -= 1; }
    const pad = (high - low) * 0.06;
    low -= pad;
    high += pad;

    const left = 54, right = 10, top = 10, bottom = 24;
    const plotW = width - left - right;
    const plotH = height - top - bottom;
    const x = (i) => left + (i / Math.max(equity.length - 1, 1)) * plotW;
    const y = (v) => top + (1 - (v - low) / (high - low)) * plotH;

    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#718096";
    ctx.strokeStyle = "#eef1f5";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const v = low + ((high - low) * i) / 4;
        const yy = y(v);
        ctx.beginPath();
        ctx.moveTo(left, yy);
        ctx.lineTo(width - right, yy);
        ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(Math.round(v).toLocaleString(), left - 6, yy + 3);
    }

    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(left, y(start));
    ctx.lineTo(width - right, y(start));
    ctx.stroke();
    ctx.setLineDash([]);

    const finalValue = equity[equity.length - 1];
    ctx.strokeStyle = finalValue >= start ? "#16a34a" : "#dc2626";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    equity.forEach((v, i) => {
        if (i === 0) ctx.moveTo(x(i), y(v));
        else ctx.lineTo(x(i), y(v));
    });
    ctx.stroke();

    ctx.fillStyle = "#718096";
    ctx.textAlign = "left";
    ctx.fillText(String(first).slice(0, 10), left, height - 7);
    ctx.textAlign = "right";
    ctx.fillText(String(last).slice(0, 10), width - right, height - 7);
}

window.addEventListener("resize", drawEquity);


/* ---------- startup ---------- */

async function init() {
    $("strategyCode").value = TEMPLATE;
    const status = document.querySelector(".data-status small");

    try {
        await loadStrategies();
        if (status) status.textContent = "Server connected";
    } catch (error) {
        if (status) status.textContent = "Server offline";
        showToast("Start the server: python -m src.server");
    }
}

init();
