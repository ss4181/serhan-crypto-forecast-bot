"use strict";

// Use DOM text nodes: data must never be interpreted as HTML or executable code.
const el = id => document.getElementById(id);
const num = value => typeof value === "number" && Number.isFinite(value) ? value : null;
const number = value => num(value) === null ? "—" : new Intl.NumberFormat("tr-TR", {maximumFractionDigits: 2}).format(value);
const price = value => num(value) === null ? "—" : "$" + new Intl.NumberFormat("en-US", {maximumSignificantDigits: 8}).format(value);
const pct = value => num(value) === null ? "—" : "%" + number(value * 100);
const date = value => {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.valueOf()) ? "—" : d.toLocaleString("tr-TR", {timeZone: "Europe/Istanbul"});
};
let signals = [];
function cell(row, text, className = "") {
  const td = document.createElement("td");
  td.textContent = String(text ?? "—");
  td.className = className;
  row.appendChild(td);
  return td;
}
function render() {
  const q = el("search").value.trim().toUpperCase();
  const kind = el("kind").value;
  const status = el("status").value;
  const rows = signals.filter(x => (!q || String(x.symbol || "").includes(q)) && (!kind || x.kind === kind) && (!status || x.status === status));
  const tbody = el("rows");
  tbody.replaceChildren();
  el("result-count").textContent = `${rows.length} / ${signals.length} kayıt gösteriliyor`;
  rows.forEach(x => {
    const row = document.createElement("tr");
    const direction = String(x.direction || "—");
    const colour = direction.includes("YUKARI") ? "up" : /AŞAĞI|ASAGI/.test(direction) ? "down" : "";
    const type = x.kind === "scalp-bracket" ? "Dinamik TP/SL" : x.kind === "scalp-target" ? "%2/%3/%5" : "Model";
    const levels = num(x.targetPercent) === null ? "—" : `%${number(x.targetPercent)}${num(x.stopPercent) === null ? "" : " / %" + number(x.stopPercent)}${num(x.targetPrice) === null ? "" : " → " + price(x.targetPrice)}`;
    const forecast = num(x.successProbability) !== null ? `${pct(x.successProbability)} / ${number(x.expectedNetBps)} bps` : typeof x.probabilityUp === "number" ? `↑ ${pct(x.probabilityUp)} / ↓ ${pct(x.probabilityDown)}` : "—";
    cell(row, date(x.sourceTimeMs));
    cell(row, type);
    cell(row, x.symbol);
    cell(row, direction, colour);
    cell(row, x.strategy || "—");
    cell(row, number(x.score));
    cell(row, `${pct(x.qualityPercentile)} / ${x.confidence || "—"}`);
    cell(row, forecast);
    cell(row, Array.isArray(x.families) ? x.families.join("+") : "—");
    cell(row, price(x.sourcePrice));
    cell(row, levels);
    cell(row, date(x.touchTimeMs));
    cell(row, x.status, x.success === true ? "hit" : x.success === false ? "miss" : "pending");
    cell(row, x.notified ? "Gönderildi" : "Sessiz");
    cell(row, number(x.netBps));
    tbody.appendChild(row);
  });
  if (!rows.length) {
    const row = document.createElement("tr");
    cell(row, "Bu filtrelere uygun kayıt yok.").colSpan = 15;
    tbody.appendChild(row);
  }
}
async function load() {
  try {
    const response = await fetch("scalp-data.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema !== "trade3-signal-dashboard-v1" || !Array.isArray(data.signals)) throw new Error("Geçersiz veri");
    signals = data.signals.filter(x => x && typeof x === "object");
    const s = data.summary || {};
    el("updated").textContent = `Son yayın: ${date(data.generatedAtUtc)} • Son sinyal: ${date(data.latestSignalAtUtc)}`;
    const stale = !data.generatedAtUtc || !Number.isFinite(Date.parse(data.generatedAtUtc)) || Date.now() - Date.parse(data.generatedAtUtc) > 6 * 3600000;
    const fresh = data.sourceStatus === "fresh" && !stale;
    el("freshness").className = `freshness ${fresh ? "fresh" : "stale"}`;
    el("freshness").textContent = fresh ? "Son yayın güncel" : "Eski veya eksik veri — yayın saatini kontrol edin";
    for (const [id, key] of [["total", "signalCount"], ["settled", "settledCount"], ["pending", "pendingCount"]]) el(id).textContent = number(s[key]);
    for (const level of [2, 3, 5]) {
      const v = (s.targetLevels || {})[String(level)];
      el("target-" + level).textContent = v ? `${number(v.hits)} dokundu · ${number(v.misses)} süresi doldu · ${number(v.pending)} bekliyor` : "Veri yok";
    }
    el("bracket-rate").textContent = `${number(s.scalpBracketWins)} / ${number(s.settledScalpBracketCount)} sonuç`;
    el("notified-rate").textContent = `${number(s.notifiedScalpTargetHits)} / ${number(s.notifiedScalpTargetCount)} kademe (2/3/5)`;
    render();
  } catch (error) {
    el("updated").textContent = "Veri yüklenemedi. Sayfayı yenileyin veya daha sonra tekrar deneyin.";
    el("freshness").className = "freshness stale";
    el("freshness").textContent = "Veri alınamadı";
    signals = [];
    render();
  }
}
for (const id of ["search", "kind", "status"]) el(id).addEventListener("input", render);
load();
