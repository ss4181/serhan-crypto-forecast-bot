const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'docs/scalp.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'docs/scalp.js'), 'utf8');
assert.equal((html.match(/<\/html>/g) || []).length, 1, 'one complete HTML document');
assert.equal(html.trimEnd().endsWith('</html>'), true, 'no trailing script fragments');
assert.match(html, /<script src="scalp.js" defer><\/script>/);
new vm.Script(js); // A syntax error must fail the build, even with an empty dataset.
class Element {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; this.listeners = {}; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  addEventListener(event, callback) { this.listeners[event] = callback; }
}
async function page(payload, ok = true) {
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const document = {getElementById: id => elements.get(id), createElement: () => new Element()};
  vm.runInNewContext(js, {document, Intl, Date, console, fetch: async () => ({ok, status: ok ? 200 : 503, json: async () => payload})});
  await new Promise(setImmediate);
  return elements;
}
async function main() {
  const payload = {schema: 'trade3-signal-dashboard-v1', sourceStatus:'fresh', generatedAtUtc: new Date().toISOString(), summary: {targetLevels: {'5':{hits:1, misses:2, pending:3}}}, signals: [
    {kind:'scalp-target', symbol:'SOLUSDT', notified:true, direction:'YUKARI', status:'HEDEF ULAŞTI', success:true, targetPercent:5, targetPrice:105, sourcePrice:100, touchTimeMs:1700000000000},
    {kind:'regular', symbol:'BTCUSDT', notified:true, direction:'ASAGI', probabilityUp:0.4, probabilityDown:0.6, status:'BEKLEMEDE'},
    {kind:'scalp-bracket', symbol:'<script>bad</script>', notified:true, direction:null, status:'STOP'},
    null,
  ]};
  let p = await page(payload);
  assert.equal(p.get('rows').children.length, 3);
  assert.equal(p.get('rows').children[0].children.length, 15);
  assert.match(p.get('rows').children[0].children[10].textContent, /105/);
  assert.notEqual(p.get('rows').children[0].children[11].textContent, '—');
  assert.equal(p.get('rows').children[2].children[2].textContent, '<script>bad</script>');
  assert.match(p.get('target-5').textContent, /1 dokundu/);
  assert.equal(p.get('audience').value, 'notified');
  assert.match(p.get('touch-cohorts').children[0].children[0].textContent, /bulunmuyor/);
  p.get('search').value='SOL'; p.get('search').listeners.input();
  assert.equal(p.get('rows').children.length, 1);
  p.get('search').value=''; p.get('kind').value='regular'; p.get('kind').listeners.input();
  assert.equal(p.get('rows').children[0].children[2].textContent, 'BTCUSDT');
  p.get('status').value='STOP'; p.get('status').listeners.input();
  assert.equal(p.get('rows').children[0].children[0].colSpan, 15);
  p = await page({...payload, signals:[]});
  assert.equal(p.get('rows').children[0].children[0].colSpan, 15);
  p = await page({}, false);
  assert.equal(p.get('freshness').textContent, 'Veri alınamadı');
  p = await page({signals:[]});
  assert.equal(p.get('freshness').textContent, 'Veri alınamadı');
  const measured = {...payload, measurements:{historyComplete:true, audiences:{notified:[
    {kind:'scalp-target', targetPercent:2, horizonHours:24, hits:1, resolvedCount:2, hitRate:.5, openCount:1, earlyHits:1, unresolvedCount:0, unknownDeadlineCount:0},
    {kind:'scalp-bracket', horizonHours:1, hits:1, stops:2, timeExits:0, resolvedCount:3, hitRate:1/3, positiveNetRate:0, meanNetBps:-25, openCount:0, unresolvedCount:0, unknownDeadlineCount:0}
  ], muted:[], all:[]}}};
  p = await page(measured);
  assert.equal(p.get('touch-cohorts').children[0].children[3].textContent, '%50');
  assert.match(p.get('bracket-cohorts').children[0].children[5].textContent, /-25/);
  p.get('audience').value='muted'; p.get('audience').listeners.input();
  assert.equal(p.get('rows').children[0].children[0].colSpan, 15);
  assert.match(p.get('touch-cohorts').children[0].children[0].textContent, /kayıt yok/);
  p = await page({...payload, signals:[{kind:'regular', status:'HEDEF', notified:true}, {kind:'regular', status:'SURE', notified:true}]});
  p.get('status').value='TARGET'; p.get('status').listeners.input();
  assert.equal(p.get('rows').children.length, 1);
  assert.equal(p.get('rows').children[0].children[12].textContent, 'HEDEF ÖNCE');
  p.get('status').value='TIME_EXIT'; p.get('status').listeners.input();
  assert.equal(p.get('rows').children.length, 1);
  assert.equal(p.get('rows').children[0].children[12].textContent, 'SÜRE SONU');
  console.log('Dashboard: syntax, HTML structure, rows, filters, missing fields, error and empty states passed.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
