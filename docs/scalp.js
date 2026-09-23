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
const localDateKey = value => {
  const stamp = num(value);
  if (stamp === null) return "";
  const parts = new Intl.DateTimeFormat("en-CA", {timeZone: "Europe/Istanbul", year: "numeric", month: "2-digit", day: "2-digit"}).formatToParts(new Date(stamp));
  const fields = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return `${fields.year}-${fields.month}-${fields.day}`;
};
let signals = [];
let measurements = null;
const canonicalStatus = status => ({HEDEF: "TARGET", SURE: "TIME_EXIT", DATA_MISSING: "VERİ EKSİK"}[status] || status);
const statusText = status => ({TARGET: "HEDEF ÖNCE", HEDEF: "HEDEF ÖNCE", TIME_EXIT: "SÜRE SONU", SURE: "SÜRE SONU", DATA_MISSING: "VERİ EKSİK", DIRECTION_HIT: "YÖN DOĞRU", DIRECTION_MISS: "YÖN YANLIŞ"}[status] || status);
function cell(row, text, className = "") {
  const td = document.createElement("td");
  td.textContent = String(text ?? "—");
  td.className = className;
  row.appendChild(td);
  return td;
}
function render() {
  const q = el("search").value.trim().toUpperCase();
  const family = el("family").value;
  const minSuccessRaw = el("minimum-success").value;
  const minSuccess = minSuccessRaw === "" ? null : Number(minSuccessRaw) / 100;
  const kind = el("kind").value;
  const direction = el("direction").value;
  const regime = el("regime").value;
  const strategy = el("strategy").value;
  const policy = el("policy").value;
  const fromDate = el("date-from").value;
  const toDate = el("date-to").value;
  const status = el("status").value;
  const audience = el("audience").value;
  const rows = signals.filter(x => {
    const day = localDateKey(x.sourceTimeMs);
    return (audience === "all" || !!x.notified === (audience === "notified"))
      && (!q || String(x.symbol || "").toUpperCase().includes(q))
      && (!family || (Array.isArray(x.families) && x.families.includes(family)))
      && (minSuccess === null || (num(x.cohortHitRate) !== null && x.cohortHitRate >= minSuccess))
      && (!kind || x.kind === kind)
      && (!direction || x.direction === direction)
      && (!regime || x.regimeState === regime)
      && (!strategy || x.strategy === strategy)
      && (!policy || (x.policyVersion || "legacy") === policy)
      && (!fromDate || (day && day >= fromDate))
      && (!toDate || (day && day <= toDate))
      && (!status || canonicalStatus(x.status) === status);
  });
  const tbody = el("rows");
  tbody.replaceChildren();
  el("result-count").textContent = `${rows.length} / ${signals.length} kayıt gösteriliyor`;
  rows.forEach(x => {
    const row = document.createElement("tr");
    const direction = String(x.direction || "—");
    const colour = direction.includes("YUKARI") ? "up" : /AŞAĞI|ASAGI/.test(direction) ? "down" : "";
    const type = x.kind === "scalp-bracket" ? "Dinamik TP/SL" : x.kind === "scalp-target" ? "%2/%3/%5" : x.kind === "scalp-forward" ? "15/30/60 dk ileri-test" : "Model";
    const levels = num(x.targetPercent) === null ? "—" : `%${number(x.targetPercent)}${num(x.stopPercent) === null ? "" : " / %" + number(x.stopPercent)}${num(x.targetPrice) === null ? "" : " → " + price(x.targetPrice)}`;
    const forecast = num(x.successProbability) !== null ? `${pct(x.successProbability)} / ${number(x.expectedNetBps)} bps` : typeof x.probabilityUp === "number" ? `↑ ${pct(x.probabilityUp)} / ↓ ${pct(x.probabilityDown)}` : "—";
    cell(row, date(x.sourceTimeMs));
    cell(row, num(x.detectionToDeliverySeconds) === null ? "—" : `${number(x.detectionToDeliverySeconds)} sn`);
    cell(row, type);
    cell(row, x.symbol);
    cell(row, direction, colour);
    cell(row, x.regimeState || "—");
    cell(row, x.strategy || "—");
    cell(row, number(x.score));
    cell(row, `${pct(x.qualityPercentile)} / ${x.confidence || "—"}${num(x.cohortHitRate) === null ? "" : ` · gerçekleşen ${pct(x.cohortHitRate)} (n=${number(x.cohortResolvedCount)})`}`);
    cell(row, forecast);
    cell(row, Array.isArray(x.families) ? x.families.join("+") : "—");
    cell(row, price(x.sourcePrice));
    cell(row, levels);
    cell(row, date(x.touchTimeMs));
    const outcomeCell = cell(row, statusText(x.status), x.success === true ? "hit" : x.success === false ? "miss" : "pending");
    if (x.missingReason) outcomeCell.title = `Neden: ${x.missingReason}`;
    cell(row, x.notified ? "Gönderildi" : "Sessiz");
    cell(row, number(x.netBps));
    cell(row, x.policyVersion || "legacy");
    tbody.appendChild(row);
  });
  if (!rows.length) {
    const row = document.createElement("tr");
    cell(row, "Bu filtrelere uygun kayıt yok.").colSpan = 18;
    tbody.appendChild(row);
  }
}
function setFilterOptions(id, title, values) {
  const select = el(id), previous = select.value;
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = ""; all.textContent = title; select.appendChild(all);
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value; option.textContent = value; select.appendChild(option);
  }
  select.value = values.includes(previous) ? previous : "";
}
function renderMeasurements() {
  const touch = el("touch-cohorts"), bracket = el("bracket-cohorts"), forward = el("forward-cohorts");
  touch.replaceChildren(); bracket.replaceChildren(); forward.replaceChildren();
  const cohorts = measurements?.audiences?.[el("audience").value];
  for (const x of Array.isArray(cohorts) ? cohorts : []) {
    const row = document.createElement("tr");
    const hours = num(x.horizonHours) === null ? "Bilinmiyor" : `${number(x.horizonHours)} saat`;
    const missing = (num(x.unresolvedCount) || 0) + (num(x.unknownDeadlineCount) || 0);
    if (x.kind === "scalp-target") {
      cell(row, `%${number(x.targetPercent)}`); cell(row, hours);
      cell(row, `${x.symbol || "—"} / ${x.family || "—"}`);
      cell(row, `${number(x.hits)} / ${number(x.resolvedCount)}`);
      cell(row, pct(x.hitRate));
      cell(row, rollingText(x.rolling));
      cell(row, `${number(x.openCount)} (${number(x.earlyHits)} erken dokunuş)`);
      cell(row, number(missing), missing ? "pending" : "");
      cell(row, `${x.direction || "UNKNOWN"} / ${x.regime || "UNKNOWN"} / ${x.strategy || "Bilinmiyor"} / ${x.policyVersion || "legacy"}`);
      cell(row, number(x.distinctUtcDays));
      cell(row, Array.isArray(x.hitRateWilson95) ? `${pct(x.hitRateWilson95[0])}–${pct(x.hitRateWilson95[1])}` : "—");
      touch.appendChild(row);
    } else if (x.kind === "scalp-bracket") {
      cell(row, hours); cell(row, `${x.symbol || "—"} / ${x.family || "—"}`); cell(row, number(x.resolvedCount));
      cell(row, `${number(x.hits)} / ${number(x.stops)} / ${number(x.timeExits)}`);
      cell(row, pct(x.hitRate)); cell(row, pct(x.positiveNetRate));
      cell(row, num(x.meanNetBps) === null ? "—" : `${number(x.meanNetBps)} bps`, x.meanNetBps > 0 ? "hit" : x.meanNetBps < 0 ? "miss" : "");
      cell(row, rollingText(x.rolling));
      cell(row, `${number(x.openCount)} / ${number(missing)}${x.missingNetCount ? ` · ${number(x.missingNetCount)} net eksik` : ""}`);
      cell(row, `${x.direction || "UNKNOWN"} / ${x.regime || "UNKNOWN"} / ${x.strategy || "Bilinmiyor"} / ${x.policyVersion || "legacy"}`);
      cell(row, number(x.distinctUtcDays));
      cell(row, Array.isArray(x.hitRateWilson95) ? `${pct(x.hitRateWilson95[0])}–${pct(x.hitRateWilson95[1])}` : "—");
      bracket.appendChild(row);
    } else if (x.kind === "scalp-forward") {
      cell(row, `${number(x.horizonHours * 60)} dk`);
      cell(row, `${x.symbol || "—"} / ${x.family || "—"}`);
      cell(row, `${number(x.hits)} / ${number(x.resolvedCount)}`);
      cell(row, pct(x.hitRate));
      cell(row, rollingText(x.rolling));
      cell(row, `${number(x.openCount)} / ${number(missing)}`);
      cell(row, `${x.regime || "UNKNOWN"} / ${x.policyVersion || "legacy"}`);
      cell(row, number(x.distinctUtcDays));
      cell(row, Array.isArray(x.hitRateWilson95) ? `${pct(x.hitRateWilson95[0])}–${pct(x.hitRateWilson95[1])}` : "—");
      forward.appendChild(row);
    }
  }
  for (const [table, columns] of [[touch, 11], [bracket, 12], [forward, 9]]) {
    if (!table.children.length) {
      const row = document.createElement("tr");
      cell(row, Array.isArray(cohorts) ? "Bu kapsamda ölçülecek kayıt yok." : "Bu yayında karşılaştırılabilir ölçümler bulunmuyor.").colSpan = columns;
      table.appendChild(row);
    }
  }
  el("measurement-note").textContent = measurements?.historyComplete === false
    ? "Geçmiş kayıt sınırına ulaşıldı; eksik geçmişten oran üretilmiyor."
    : "Gruplar yön, rejim ve stratejiye ayrılır. Oranlar yalnız takip süresi dolmuş kayıtları kapsar; aralık %95 Wilson aralığıdır ve aynı gün/coin bağımlılığını düzeltmez. —: hesaplanamıyor.";
}
function rollingText(rolling) {
  if (!rolling) return "—";
  return [10, 50, 100].map(size => {
    const x = rolling[String(size)] || {};
    const ci = Array.isArray(x.wilson95) ? ` (${pct(x.wilson95[0])}–${pct(x.wilson95[1])})` : "";
    return `${size}: ${number(x.count)} · ${pct(x.rate)}${ci}`;
  }).join(" | ");
}
async function load() {
  try {
    const response = await fetch("scalp-data.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema !== "trade3-signal-dashboard-v1" || !Array.isArray(data.signals)) throw new Error("Geçersiz veri");
    signals = data.signals.filter(x => x && typeof x === "object");
    setFilterOptions("family", "Tüm aileler", [...new Set(signals.flatMap(x => Array.isArray(x.families) ? x.families : []).filter(Boolean))].sort());
    setFilterOptions("direction", "Tüm yönler", [...new Set(signals.map(x => x.direction).filter(Boolean))].sort());
    setFilterOptions("regime", "Tüm rejimler", [...new Set(signals.map(x => x.regimeState).filter(Boolean))].sort());
    setFilterOptions("strategy", "Tüm stratejiler", [...new Set(signals.map(x => x.strategy).filter(Boolean))].sort());
    setFilterOptions("policy", "Tüm politikalar", [...new Set(signals.map(x => x.policyVersion || "legacy"))].sort());
    measurements = data.measurements || null;
    const s = data.summary || {};
    el("updated").textContent = `Son yayın: ${date(data.generatedAtUtc)} • Son sinyal: ${date(data.latestSignalAtUtc)}`;
    const stale = !data.generatedAtUtc || !Number.isFinite(Date.parse(data.generatedAtUtc)) || Date.now() - Date.parse(data.generatedAtUtc) > 6 * 3600000;
    const fresh = data.sourceStatus === "fresh" && !stale;
    el("freshness").className = `freshness ${fresh ? "fresh" : "stale"}`;
    el("freshness").textContent = fresh ? "Son yayın güncel" : "Eski veya eksik veri — yayın saatini kontrol edin";
    for (const [id, key] of [["total", "signalCount"], ["settled", "settledCount"], ["pending", "pendingCount"]]) el(id).textContent = number(s[key]);
    const quarantined = Number(s.quarantinedRecordCount) || 0;
    const quarantineNote = el("quarantine-note");
    quarantineNote.hidden = quarantined < 1;
    quarantineNote.textContent = quarantined
      ? `${number(quarantined)} bozuk bekleyen kayıt karantinaya alındı; başarı hesabına katılmıyor. Sunucu kayıtları incelenmeli.`
      : "";
    for (const level of [2, 3, 5]) {
      const v = (s.targetLevels || {})[String(level)];
      el("target-" + level).textContent = v ? `${number(v.hits)} dokundu · ${number(v.misses)} süresi doldu · ${number(v.pending)} bekliyor` : "Veri yok";
    }
    renderMeasurements();
    render();
  } catch (error) {
    el("updated").textContent = "Veri yüklenemedi. Sayfayı yenileyin veya daha sonra tekrar deneyin.";
    el("freshness").className = "freshness stale";
    el("freshness").textContent = "Veri alınamadı";
    signals = [];
    measurements = null;
    renderMeasurements();
    render();
  }
}
for (const id of ["search", "family", "minimum-success", "kind", "direction", "regime", "strategy", "policy", "status", "date-from", "date-to"]) el(id).addEventListener(id.startsWith("date-") ? "change" : "input", render);
el("audience").value = "all";
el("audience").addEventListener("input", () => { renderMeasurements(); render(); });
load();
