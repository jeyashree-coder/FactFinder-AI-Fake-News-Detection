const resultNode = document.getElementById("result");
const historyNode = document.getElementById("history");
const bbcNewsNode = document.getElementById("bbc-news");


function setActiveTab(tabName) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}-panel`);
  });
}

function renderResult(payload, preview = "") {
  const isReal = payload.prediction === "Real";
  resultNode.className = `result ${isReal ? "real" : "fake"}`;
  resultNode.innerHTML = `
    <h3 class="verdict">${payload.prediction}</h3>
    <p class="confidence">Confidence: ${(payload.confidence * 100).toFixed(2)}%</p>
    <div style="background: rgba(0,0,0,0.05); padding: 12px; margin: 15px 0; border-left: 4px solid ${isReal ? '#2e7d32' : '#c62828'}; font-style: italic; border-radius: 4px;">
        <strong>✨ AI Explanation:</strong> ${payload.interpretation}
    </div>
    <p><strong>Random Forest:</strong> ${(payload.details.random_forest_real_probability * 100).toFixed(2)}% real</p>
    <p><strong>XGBoost:</strong> ${(payload.details.xgboost_real_probability * 100).toFixed(2)}% real</p>
    <p><strong>Training data:</strong> ${payload.details.metadata.training_size} samples.</p>
    ${preview ? `<p><strong>Extracted preview:</strong> ${preview}</p>` : ""}
  `;
}

function renderEmptyState(node, text) {
  node.innerHTML = `<div class="item"><p>${text}</p></div>`;
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

async function loadHistory() {
  const response = await fetch("/history");
  const data = await response.json();
  if (!data.history.length) {
    renderEmptyState(historyNode, "No prediction history yet.");
    return;
  }
  historyNode.innerHTML = data.history.map((item) => `
    <article class="item">
      <h3>${item.prediction} • ${(item.confidence * 100).toFixed(2)}%</h3>
      <p>${item.input_value.slice(0, 180)}</p>
      <p class="meta">${item.input_type.toUpperCase()} • ${new Date(item.created_at).toLocaleString()}</p>
    </article>
  `).join("");
}

async function loadBBCNews() {
  const response = await fetch("/today-news");
  const data = await response.json();
  if (!data.news || !data.news.length) {
    renderEmptyState(bbcNewsNode, "No BBC news collected today.");
    return;
  }
  bbcNewsNode.innerHTML = data.news.map((item) => `
    <article class="item">
      <h3><a href="${item.url}" target="_blank">${item.title}</a></h3>
      <p>${item.content.slice(0, 180)}...</p>
      <p class="meta">Collected: ${new Date(item.collected_at).toLocaleString()}</p>
    </article>
  `).join("");
}


document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});

document.getElementById("predict-text-btn").addEventListener("click", async () => {
  const text = document.getElementById("news-text").value.trim();
  if (!text) {
    window.alert("Please paste some news text first.");
    return;
  }
  try {
    resultNode.className = "result empty";
    resultNode.innerHTML = "<p>Analyzing text...</p>";
    const data = await postJSON("/predict-text", { text });
    renderResult(data);
    loadHistory();
  } catch (error) {
    resultNode.className = "result empty";
    resultNode.innerHTML = `<p>${error.message}</p>`;
  }
});

document.getElementById("predict-url-btn").addEventListener("click", async () => {
  let url = document.getElementById("news-url").value.trim();
  if (!url) {
    window.alert("Please enter a URL.");
    return;
  }
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    url = "https://" + url;
  }
  try {
    resultNode.className = "result empty";
    resultNode.innerHTML = "<p>Fetching and analyzing article...</p>";
    const data = await postJSON("/predict-url", { url });
    renderResult(data, data.extracted_preview);
    loadHistory();
  } catch (error) {
    resultNode.className = "result empty";
    resultNode.innerHTML = `<p>${error.message}</p>`;
  }
});


document.getElementById("load-history-btn").addEventListener("click", loadHistory);
document.getElementById("collect-bbc-btn").addEventListener("click", async () => {
  const btn = document.getElementById("collect-bbc-btn");
  const originalText = btn.textContent;
  btn.textContent = "Collecting...";
  btn.disabled = true;
  try {
    const data = await postJSON("/collect-bbc-news", {});
    alert(`Collected ${data.inserted} new articles. Checked ${data.checked_urls} urls.`);
    loadBBCNews();
  } catch (error) {
    alert("Error collecting BBC News: " + error.message);
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
});

loadHistory();
loadBBCNews();
