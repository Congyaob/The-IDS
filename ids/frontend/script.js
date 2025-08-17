let lastId = 0;
let timer = null;
let paused = false;

const feedBody = document.getElementById("feedBody");
const banner = document.getElementById("banner");
const statusEl = document.getElementById("status");
const pauseBtn = document.getElementById("pauseBtn");
const resumeBtn = document.getElementById("resumeBtn");

pauseBtn.onclick = () => {
    paused = true;
    clearInterval(timer);
    pauseBtn.disabled = true;
    resumeBtn.disabled = false;
    statusEl.textContent = "Paused";
};

resumeBtn.onclick = () => {
    paused = false;
    startPolling();
    pauseBtn.disabled = false;
    resumeBtn.disabled = true;
    statusEl.textContent = "Listening...";
};

function startPolling() {
    timer = setInterval(fetchLogs, 2000);
    fetchLogs();
}

async function fetchLogs() {
    if (paused) return;
    try {
        const res = await fetch(`/logs?since=${lastId}&limit=200`);
        if (!res.ok) return;
        const data = await res.json();
        const items = data.items || [];

        if (items.length) {
            lastId = data.last_id || lastId;
            for (const it of items) {
                appendRow(it);
                if (it.prediction !== "Class0") {
                    showAlert(`Suspected attack detected：${it.prediction_label}（Confidence ${(it.max_conf * 100).toFixed(2)}%）`);
                }
            }
        }
    } catch (e) {
        console.error("fetch logs failed:", e);
    }
}

function appendRow({ ts, prediction, prediction_label, max_conf }) {
    const tr = document.createElement("tr");
    const benign = prediction === "Class0";

    tr.innerHTML = `
    <td>${ts}</td>
    <td class="${benign ? 'benign' : 'attack'}">${prediction_label}</td>
    <td>${(max_conf * 100).toFixed(2)}%</td>
    <td>${benign ? 'Normal traffic' : 'Please check the abnormal behavior of the connection/host'}</td>
  `;
    feedBody.appendChild(tr);

    const MAX_ROWS = 500;
    while (feedBody.rows.length > MAX_ROWS) {
        feedBody.deleteRow(0);
    }

    tr.scrollIntoView({ block: "end" });
}

let bannerTimer = null;
function showAlert(msg) {
    banner.textContent = msg;
    banner.classList.remove("hidden");
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(() => {
        banner.classList.add("hidden");
    }, 4000);
}

startPolling();

