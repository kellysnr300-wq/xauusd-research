const navItems = document.querySelectorAll(".nav-item");
const tabs = document.querySelectorAll(".tab");
const strategySelectors = document.querySelectorAll(".strategy-selector");

const editor = document.getElementById("strategy-editor");
const newStrategyButton = document.getElementById("newStrategy");
const closeEditorButton = document.getElementById("closeEditor");

const saveDraftButton = document.getElementById("saveDraft");
const saveStrategyButton = document.getElementById("saveStrategy");

const strategyName = document.getElementById("strategyName");
const strategyDescription = document.getElementById("strategyDescription");
const strategyCode = document.getElementById("strategyCode");

const strategyList = document.getElementById("strategy-list");
const toast = document.getElementById("toast");


function showTab(target) {

    navItems.forEach((item) => {
        item.classList.toggle(
            "active",
            item.dataset.tab === target
        );
    });

    tabs.forEach((tab) => {
        tab.classList.toggle(
            "active",
            tab.id === target
        );
    });

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


navItems.forEach((item) => {

    item.addEventListener("click", () => {
        showTab(item.dataset.tab);
    });

});


strategySelectors.forEach((selector) => {

    selector.addEventListener("change", (event) => {

        const selectedStrategy = event.target.value;

        strategySelectors.forEach((otherSelector) => {
            otherSelector.value = selectedStrategy;
        });

        showToast(
            `Viewing: ${selectedStrategy}`
        );

    });

});


function showToast(message) {

    toast.textContent = message;

    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 2200);

}


newStrategyButton.addEventListener("click", () => {

    editor.classList.remove("hidden");

    strategyName.focus();

    editor.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

});


closeEditorButton.addEventListener("click", () => {

    editor.classList.add("hidden");

});


saveDraftButton.addEventListener("click", () => {

    const name = strategyName.value.trim();

    if (!name) {
        showToast("Enter a strategy name first.");
        strategyName.focus();
        return;
    }

    showToast(`Draft saved: ${name}`);

});


saveStrategyButton.addEventListener("click", () => {

    const name = strategyName.value.trim();
    const description = strategyDescription.value.trim();

    if (!name) {
        showToast("Enter a strategy name first.");
        strategyName.focus();
        return;
    }

    const card = document.createElement("div");

    card.className = "strategy-card";

    card.innerHTML = `
        <div class="strategy-card-top">
            <span class="strategy-type">CUSTOM</span>
            <span class="tag draft">Draft</span>
        </div>

        <h3>${escapeHtml(name)}</h3>

        <p>
            ${escapeHtml(
                description ||
                "Custom research strategy."
            )}
        </p>

        <div class="strategy-footer">
            <span>0 experiments</span>
            <button class="text-button">Open</button>
        </div>
    `;

    strategyList.appendChild(card);

    showToast(`Strategy saved: ${name}`);

    strategyName.value = "";
    strategyDescription.value = "";

    editor.classList.add("hidden");

});


function escapeHtml(value) {

    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}


console.log("XAU/USD Research Dashboard loaded.");
