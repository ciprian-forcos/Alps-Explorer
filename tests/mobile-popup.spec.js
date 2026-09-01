#!/usr/bin/env node
/**
 * Mobile popup + chrome checks for Alps Explorer.
 * Serves the LOCAL tree (not github.io). Run:
 *   node tests/mobile-popup.spec.js
 *
 * Needs playwright-core + Chromium. Override with PLAYWRIGHT_CORE and
 * PLAYWRIGHT_CHROMIUM if they are not on NODE_PATH / not the nest install.
 */
'use strict';

const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');

function loadPlaywright() {
  if (process.env.PLAYWRIGHT_CORE) return require(process.env.PLAYWRIGHT_CORE);
  try { return require('playwright-core'); } catch (_) { /* nest fallback */ }
  return require(path.resolve(__dirname, '../../../.scratch/alps/node_modules/playwright-core'));
}

const { chromium } = loadPlaywright();
const OUT_JSON = process.env.ALPS_TEST_OUT ||
  path.join(os.tmpdir(), 'alps-popup-test-results.json');
const ROOT = path.resolve(__dirname, '..');

function chromePath() {
  if (process.env.PLAYWRIGHT_CHROMIUM) return process.env.PLAYWRIGHT_CHROMIUM;
  const guesses = [
    path.join(process.env.LOCALAPPDATA || '', 'ms-playwright/chromium-1223/chrome-win64/chrome.exe'),
    path.join(process.env.HOME || '', '.cache/ms-playwright/chromium-1223/chrome-linux/chrome')
  ];
  return guesses.find((p) => p && fs.existsSync(p));
}

const results = [];
function check(name, pass, detail) {
  results.push({ name, pass: !!pass, detail: detail || '' });
  console.log((pass ? 'PASS  ' : 'FAIL  ') + name + (detail ? '  — ' + detail : ''));
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function mimeFor(file) {
  const ext = path.extname(file).toLowerCase();
  return {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
  }[ext] || 'application/octet-stream';
}

function startServer(root) {
  const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      let urlPath = '/index.html';
      try {
        urlPath = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
      } catch (_) { /* keep default */ }
      if (urlPath === '/') urlPath = '/index.html';
      const file = path.normalize(path.join(rootWithSep, urlPath.replace(/^[/\\]+/, '')));
      if (!file.startsWith(rootWithSep)) {
        res.writeHead(403);
        res.end('forbidden');
        return;
      }
      fs.readFile(file, (err, data) => {
        if (err) {
          res.writeHead(404, { 'Content-Type': 'text/plain' });
          res.end('not found');
          return;
        }
        res.writeHead(200, { 'Content-Type': mimeFor(file), 'Cache-Control': 'no-store' });
        res.end(data);
      });
    });
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, port, origin: 'http://127.0.0.1:' + port });
    });
    server.on('error', reject);
  });
}

function writeReport(extra) {
  const passed = results.filter((r) => r.pass).length;
  const payload = {
    ranAt: new Date().toISOString(),
    origin: extra.origin || '',
    head: extra.head || '',
    passed,
    total: results.length,
    score: passed + '/' + results.length,
    results
  };
  fs.mkdirSync(path.dirname(OUT_JSON), { recursive: true });
  fs.writeFileSync(OUT_JSON, JSON.stringify(payload, null, 2));
  return payload;
}

async function waitForMap(page) {
  await page.waitForFunction(() => {
    return typeof L === 'object' && typeof map !== 'undefined' &&
      Array.isArray(items) && items.length > 0 &&
      document.querySelector('#map.leaflet-container');
  }, { timeout: 30000 });
  await page.evaluate(() => { try { map.invalidateSize({ animate: false }); } catch (_) {} });
  await sleep(400);
}

async function installPopupLog(page) {
  await page.evaluate(() => {
    window.__popupLog = [];
    if (window.__popupLogBound) return;
    window.__popupLogBound = true;
    map.on('popupopen', function () {
      const name = (document.querySelector('.leaflet-popup .popup-name') || {}).textContent || '';
      window.__popupLog.push({ t: Date.now(), ev: 'open', name: name.trim() });
    });
    map.on('popupclose', function () {
      window.__popupLog.push({ t: Date.now(), ev: 'close' });
    });
  });
}

async function pickTargets(page) {
  return page.evaluate(() => {
    function fold(s) {
      return String(s).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    }
    function isolationM(it) {
      let min = Infinity;
      for (let i = 0; i < items.length; i++) {
        const other = items[i];
        if (other === it) continue;
        const d = map.distance([it.lat, it.lng], [other.lat, other.lng]);
        if (d < min) min = d;
      }
      return min;
    }
    function mostIsolated(kind) {
      const pool = items.filter(function (it) { return it.kind === kind; });
      let best = pool[0];
      let bestD = -1;
      for (let i = 0; i < pool.length; i++) {
        const d = isolationM(pool[i]);
        if (d > bestD) { bestD = d; best = pool[i]; }
      }
      return best ? { name: best.name, kind: best.kind, isolationM: Math.round(bestD) } : null;
    }
    const uniqueVillage = items.find(function (v) {
      if (v.kind !== 'village') return false;
      const needle = fold(v.name);
      return items.filter(function (i) { return i.haystack.indexOf(needle) !== -1; }).length === 1;
    });
    return {
      village: mostIsolated('village'),
      resort: mostIsolated('resort'),
      uniqueVillage: uniqueVillage ? uniqueVillage.name : (items.find(function (i) { return i.kind === 'village'; }) || {}).name
    };
  });
}

async function revealMarker(page, name) {
  return page.evaluate(async (name) => {
    const it = items.find(function (i) { return i.name === name; });
    if (!it) return { ok: false, reason: 'no item named ' + name };
    if (map.closePopup) map.closePopup();
    const zoom = 13;
    map.setView([it.lat, it.lng], zoom, { animate: false });
    if (typeof markerLayer.zoomToShowLayer === 'function' && !(it.marker && it.marker._icon)) {
      await new Promise(function (resolve) {
        try { markerLayer.zoomToShowLayer(it.marker, resolve); }
        catch (e) { resolve(); }
      });
    }
    map.invalidateSize({ animate: false });
    const icon = it.marker && it.marker._icon;
    const r = icon ? icon.getBoundingClientRect() : null;
    return {
      ok: !!(r && r.width > 0 && r.height > 0),
      name: it.name,
      kind: it.kind,
      zoom: map.getZoom(),
      clustered: !icon,
      rect: r ? { x: r.x, y: r.y, w: r.width, h: r.height } : null
    };
  }, name);
}

async function markerBoundingBox(page, name) {
  const handle = await page.evaluateHandle((n) => {
    const it = items.find(function (i) { return i.name === n; });
    return it && it.marker && it.marker._icon ? it.marker._icon : null;
  }, name);
  const el = handle.asElement();
  if (!el) return null;
  return el.boundingBox();
}

async function realTap(page, x, y) {
  // Real touch, not locator.click(). Prefer Playwright touchscreen (hasTouch
  // context); fall back to CDP if the point is reported off-screen.
  const px = Math.round(x);
  const py = Math.round(y);
  try {
    await page.touchscreen.tap(px, py);
  } catch (_) {
    const client = await page.context().newCDPSession(page);
    await client.send('Input.dispatchTouchEvent', {
      type: 'touchStart',
      touchPoints: [{ x: px, y: py, id: 1, radiusX: 8, radiusY: 8, force: 0.7 }]
    });
    await sleep(40);
    await client.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await client.detach().catch(() => {});
  }
}

async function tapPointOnMarker(page, name) {
  const box = await markerBoundingBox(page, name);
  if (!box) return { ok: false, reason: 'marker icon not in DOM for ' + name };
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;
  const hit = await page.evaluate(({ x, y, name }) => {
    const el = document.elementFromPoint(x, y);
    const icon = el && el.closest && el.closest('.leaflet-marker-icon');
    return {
      hit: !!(icon && icon.getAttribute('title') === name),
      title: icon ? icon.getAttribute('title') : (el && (el.className || el.tagName)),
      x: x,
      y: y
    };
  }, { x, y, name });
  if (!hit.hit) return { ok: false, reason: 'tap target is not the pin (' + hit.title + ')', x, y };
  await realTap(page, x, y);
  return { ok: true, x, y };
}

async function tapMarkerOrFallback(page, name) {
  let tap = await tapPointOnMarker(page, name);
  if (tap.ok) return tap;
  const fallback = await page.evaluate((n) => {
    const it = items.find(function (i) { return i.name === n; });
    if (!it || !it.marker || !it.marker._icon) return null;
    const r = it.marker._icon.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }, name);
  if (!fallback) return tap;
  await realTap(page, fallback.x, fallback.y);
  return { ok: true, x: fallback.x, y: fallback.y, reason: tap.reason };
}

// The click shield swallows clicks on the map container for 500ms after a popup
// opens. It must eat only the ghost click on the map surface -- a deliberate tap
// on a DIFFERENT pin has to still work. Walk unclustered views until we find
// scenarios where pin B is genuinely exposed (not under A's popup, not under a
// control), tap A then B inside the shield window, and require B to win.
const SHIELD_TAP_DELAY_MS = 200;

function findFairPair() {
  const boxes = [];
  items.forEach(function (it) {
    const icon = it.marker && it.marker._icon;
    if (!icon) return;
    const r = icon.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return;
    const cx = r.x + r.width / 2;
    const cy = r.y + r.height / 2;
    // keep clear of the header, the count overlay and the attribution control
    if (cx < 70 || cx > window.innerWidth - 70) return;
    if (cy < 140 || cy > window.innerHeight - 260) return;
    const el = document.elementFromPoint(cx, cy);
    const hit = el && el.closest && el.closest('.leaflet-marker-icon');
    if (hit && hit.getAttribute('title') === it.name) boxes.push({ name: it.name, cx: cx, cy: cy });
  });
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const px = Math.hypot(boxes[i].cx - boxes[j].cx, boxes[i].cy - boxes[j].cy);
      if (px >= 110) return { a: boxes[i], b: boxes[j], px: Math.round(px), boxCount: boxes.length };
    }
  }
  return { boxCount: boxes.length, icons: items.filter(function (i) { return i.marker && i.marker._icon; }).length };
}

async function secondPinScenarios(page, wanted) {
  const out = [];
  const stats = { centres: 0, pairs: 0, exposed: 0, driftedOut: 0, tapMissed: 0, shieldExpired: 0 };
  // earlier checks leave a search term and an open drawer behind; both would
  // starve this scan (a filtered render puts almost no pins on the map)
  await page.evaluate(() => {
    const side = document.getElementById('sidebar');
    if (side) side.classList.remove('open');
    const se = document.getElementById('search');
    if (se) { se.value = ''; se.dispatchEvent(new Event('input')); }
    state.resorts = true;
    state.villages = true;
    state.country = 'all';
    render();
    map.closePopup();
    window.__tapHit = null;
    if (!window.__tapHitBound) {
      window.__tapHitBound = true;
      document.addEventListener('touchstart', function (e) {
        const t = e.touches && e.touches[0];
        if (!t) return;
        const el = document.elementFromPoint(t.clientX, t.clientY);
        const icon = el && el.closest && el.closest('.leaflet-marker-icon');
        window.__tapHit = {
          title: icon ? icon.getAttribute('title') : null,
          shieldLeft: typeof suppressMapClickUntil === 'number'
            ? suppressMapClickUntil - Date.now() : null
        };
      }, true);
    }
  });
  await sleep(400);
  const total = await page.evaluate(() => items.length);
  for (let i = 0; i < total && out.length < wanted; i++) {
    await page.evaluate((idx) => {
      map.closePopup();
      const it = items[idx];
      map.setView([it.lat, it.lng], 12, { animate: false });
      map.invalidateSize({ animate: false });
    }, i);
    await sleep(500);
    stats.centres++;
    const pair = await page.evaluate(findFairPair);
    stats.lastBoxes = pair.boxCount;
    stats.lastIcons = pair.icons;
    if (!pair.a) continue;
    stats.pairs++;
    await realTap(page, pair.a.cx, pair.a.cy);
    await page.waitForSelector('.leaflet-popup', { timeout: 3000 }).catch(() => null);
    await sleep(SHIELD_TAP_DELAY_MS);
    const bNow = await page.evaluate((n) => {
      const it = items.find(function (i) { return i.name === n; });
      const icon = it && it.marker && it.marker._icon;
      if (!icon) return null;
      const r = icon.getBoundingClientRect();
      const cx = r.x + r.width / 2;
      const cy = r.y + r.height / 2;
      const el = document.elementFromPoint(cx, cy);
      const hit = el && el.closest && el.closest('.leaflet-marker-icon');
      return {
        cx: cx, cy: cy,
        exposed: !!(hit && hit.getAttribute('title') === n),
        // same tappable band findFairPair used when it picked the pair; A's
        // autoPan can slide B out of it, and a pin a few px off the viewport
        // edge is not something a tap can reach
        inBand: cx >= 70 && cx <= window.innerWidth - 70 &&
                cy >= 140 && cy <= window.innerHeight - 260
      };
    }, pair.b.name);
    // if B ended up under A's popup the user could not tap it either -- not a fair probe
    if (!bNow || !bNow.exposed) continue;
    if (!bNow.inBand) { stats.driftedOut++; continue; }
    await page.evaluate(() => { window.__tapHit = null; });
    await realTap(page, bNow.cx, bNow.cy);
    // Record what the touch actually landed on. A's popup autoPans the map, so
    // between measuring B and dispatching the touch B can slide out from under
    // the point -- that tests pan timing, not the shield. Score only the taps
    // that really hit B while the shield was still up.
    const tapHit = await page.evaluate(() => window.__tapHit);
    if (!tapHit || tapHit.title !== pair.b.name) { stats.tapMissed++; continue; }
    if (tapHit.shieldLeft === null) { stats.noShieldVar = true; continue; }
    if (tapHit.shieldLeft <= 0) { stats.shieldExpired++; continue; }
    stats.exposed++;
    await sleep(900);
    const got = await page.evaluate(() =>
      ((document.querySelector('.leaflet-popup .popup-name') || {}).textContent || '').trim());
    out.push({ a: pair.a.name, b: pair.b.name, got: got, ok: got === pair.b.name });
  }
  return { scenarios: out, stats: stats };
}

// After a pin opens a tall popup, Leaflet autoPan moves the map and a leftover
// map click (the phone bug) lands on the tiles, not the pin. Tap empty map to
// encode that: unfixed closePopupOnClick dismisses; the mobile fix must keep it.
async function tapEmptyMap(page) {
  const pt = await page.evaluate(() => {
    const mapEl = document.getElementById('map');
    if (!mapEl) return null;
    const r = mapEl.getBoundingClientRect();
    const pts = [
      [r.left + 28, r.bottom - 90],
      [r.right - 28, r.bottom - 90],
      [r.left + 28, r.top + 70],
      [r.right - 28, r.top + 70],
      [r.left + 40, r.top + r.height / 2]
    ];
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0];
      const y = pts[i][1];
      const el = document.elementFromPoint(x, y);
      if (!el) continue;
      if (el.closest('.leaflet-popup')) continue;
      if (el.closest('.leaflet-marker-icon')) continue;
      if (el.closest('.leaflet-control')) continue;
      if (el.closest('button, a, input')) continue;
      if (el.closest('#sidebar')) continue;
      if (el.closest('.map-overlay, .controls, header')) continue;
      if (!el.closest('#map')) continue;
      return { x: x, y: y };
    }
    return null;
  });
  if (!pt) return { tapped: false };
  await realTap(page, pt.x, pt.y);
  return { tapped: true, x: pt.x, y: pt.y };
}

async function popupSnapshot(page) {
  return page.evaluate(() => {
    const pop = document.querySelector('.leaflet-popup');
    if (!pop) {
      return { present: false, log: window.__popupLog || [] };
    }
    const r = pop.getBoundingClientRect();
    const w = window.innerWidth;
    const h = window.innerHeight;
    const slack = 12;
    const fully = r.left >= 0 && r.top >= 0 && r.right <= w && r.bottom <= h;
    const nearly = r.left >= -slack && r.top >= -slack && r.right <= w + slack && r.bottom <= h + slack;
    return {
      present: true,
      name: ((document.querySelector('.leaflet-popup .popup-name') || {}).textContent || '').trim(),
      type: ((document.querySelector('.leaflet-popup .popup-type') || {}).textContent || '').trim(),
      propertyLinks: document.querySelectorAll('.leaflet-popup .popup-link.property').length,
      rect: { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height },
      viewport: { w, h },
      fully,
      nearly,
      overflow: {
        left: Math.min(0, r.left),
        top: Math.min(0, r.top),
        right: Math.max(0, r.right - w),
        bottom: Math.max(0, r.bottom - h)
      },
      log: window.__popupLog || []
    };
  });
}

async function openDrawer(page) {
  const state = await page.evaluate(() => {
    const side = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebarToggle');
    const cs = toggle ? getComputedStyle(toggle) : null;
    return {
      open: !!(side && side.classList.contains('open')),
      toggleVisible: !!(cs && cs.display !== 'none' && cs.visibility !== 'hidden' && toggle.offsetParent !== null)
    };
  });
  if (state.open) return true;
  if (state.toggleVisible) {
    const box = await page.locator('#sidebarToggle').boundingBox();
    if (box) await realTap(page, box.x + box.width / 2, box.y + box.height / 2);
    await sleep(350);
  } else {
    const alt = page.locator('[aria-controls="sidebar"]').first();
    if (await alt.count()) {
      const box = await alt.boundingBox();
      if (box) await realTap(page, box.x + box.width / 2, box.y + box.height / 2);
      await sleep(350);
    }
  }
  return page.evaluate(() => {
    const side = document.getElementById('sidebar');
    if (!side) return false;
    if (side.classList.contains('open')) return true;
    const r = side.getBoundingClientRect();
    return r.width > 40 && r.left >= -1 && r.left < window.innerWidth / 2;
  });
}

(async () => {
  const { server, origin } = await startServer(ROOT);
  let browser;
  const extra = { origin, head: '' };
  try {
    extra.head = fs.existsSync(path.join(ROOT, '.git', 'HEAD'))
      ? fs.readFileSync(path.join(ROOT, '.git', 'HEAD'), 'utf8').trim()
      : '';
    const exe = chromePath();
    browser = await chromium.launch(exe ? { executablePath: exe } : {});
    const mobile = await browser.newContext({
      viewport: { width: 390, height: 844 },
      hasTouch: true,
      isMobile: true,
      deviceScaleFactor: 2
    });
    const page = await mobile.newPage();
    const jsErrors = [];
    page.on('pageerror', (err) => jsErrors.push(String(err && err.message || err)));
    page.on('console', (msg) => { if (msg.type() === 'error') jsErrors.push('console: ' + msg.text()); });
    page.setDefaultTimeout(20000);
    await page.goto(origin + '/', { waitUntil: 'load', timeout: 60000 });
    await waitForMap(page);
    await installPopupLog(page);

    const targets = await pickTargets(page);

    // ---- 5. no horizontal overflow (chrome, independent of popups) ----
    const overflow = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth
    }));
    check(
      'no horizontal overflow at 390px',
      overflow.scrollWidth <= 391,
      'scrollWidth=' + overflow.scrollWidth + ' innerWidth=' + overflow.innerWidth
    );

    // ---- 6. map is not a leftover square ----
    const mapFrac = await page.evaluate(() => {
      try { map.invalidateSize({ animate: false }); } catch (_) {}
      const el = document.getElementById('map');
      const ratio = el.clientHeight / window.innerHeight;
      return {
        mapW: el.clientWidth,
        mapH: el.clientHeight,
        innerH: window.innerHeight,
        innerW: window.innerWidth,
        ratio: ratio
      };
    });
    check(
      'map is not a leftover square (height >= 65% of viewport)',
      mapFrac.ratio >= 0.65,
      'map ' + mapFrac.mapW + 'x' + mapFrac.mapH + ' ratio=' + mapFrac.ratio.toFixed(3) +
        ' vs viewport ' + mapFrac.innerW + 'x' + mapFrac.innerH
    );

    // ---- 7. mobile filters are reachable without opening the drawer ----
    // Tester feedback 2026-09-01: with the strip hidden, the only route to the
    // filters was the header button labelled "List", and nobody found it.
    const filters = await page.evaluate(() => {
      const el = document.querySelector('.controls');
      if (!el) return { pass: false, reason: 'no .controls strip in the document' };
      const cs = getComputedStyle(el);
      const drawerOpen = document.getElementById('sidebar').classList.contains('open');
      const h = el.getBoundingClientRect().height;
      const wanted = ['[data-filter="resorts"]', '[data-filter="villages"]',
                      '[data-country="all"]', '[data-country="France"]', '[data-country="Italy"]',
                      '[data-style="street"]', '[data-style="terrain"]', '[data-style="satellite"]'];
      const missing = [];
      wanted.forEach(function (sel) {
        const b = el.querySelector(sel);
        if (!b) { missing.push(sel + ':absent'); return; }
        // scroll it into the strip's visible range, then hit-test its centre
        b.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        const r = b.getBoundingClientRect();
        if (r.width < 24 || r.height < 24) { missing.push(sel + ':too-small'); return; }
        const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        if (!hit || !el.contains(hit)) missing.push(sel + ':covered');
      });
      const visible = cs.display !== 'none' && cs.visibility !== 'hidden' && h >= 8;
      const compact = h <= window.innerHeight * 0.12;
      return {
        pass: visible && compact && !drawerOpen && missing.length === 0,
        reason: 'display=' + cs.display + ' height=' + Math.round(h) +
          ' (' + (h / window.innerHeight * 100).toFixed(1) + '% of viewport)' +
          ' drawerOpen=' + drawerOpen +
          (missing.length ? ' unreachable=' + missing.join(',') : ' all 8 controls hit-testable')
      };
    });
    check('mobile filters reachable without the drawer, strip stays compact', filters.pass, filters.reason);

    // ---- 7b. app box matches the visible viewport, no page background exposed ----
    // Tester feedback 2026-09-01: "o mare bara albastra jos" — the --ink page
    // background showing below the map because the app box outran the viewport.
    const measureFit = () => page.evaluate(() => {
      const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
      const appH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--app-h'));
      const bodyH = document.body.getBoundingClientRect().height;
      const stageBottom = document.querySelector('.stage').getBoundingClientRect().bottom;
      const htmlBg = getComputedStyle(document.documentElement).backgroundColor;
      const varSet = Number.isFinite(appH) && Math.abs(appH - vh) <= 1;
      const bodyFits = Math.abs(bodyH - vh) <= 1;
      const noGap = Math.abs(stageBottom - vh) <= 1;
      // second line of defence: if anything ever does peek through, it must not
      // be the dark --ink page background the tester saw as a blue band
      const bgNotInk = htmlBg.replace(/\s/g, '') !== 'rgb(26,26,46)';
      return {
        pass: varSet && bodyFits && noGap && bgNotInk,
        reason: '--app-h=' + (Number.isFinite(appH) ? appH : 'unset') + ' visualViewport=' + Math.round(vh) +
          ' body=' + Math.round(bodyH) + ' stageBottom=' + Math.round(stageBottom) +
          ' htmlBg=' + htmlBg
      };
    });
    const fit = await measureFit();
    check('app box equals the visible viewport (no background band)', fit.pass, fit.reason);

    // ---- 8. default basemap is Terrain ----
    const terrain = await page.evaluate(() => {
      const btns = [...document.querySelectorAll('button')];
      const terrainPressed = btns.some(function (b) {
        return /terrain/i.test(b.textContent || '') && b.getAttribute('aria-pressed') === 'true';
      });
      const dataPressed = [...document.querySelectorAll('[data-style="terrain"]')]
        .some(function (b) { return b.getAttribute('aria-pressed') === 'true'; });
      const tiles = [...document.querySelectorAll('.leaflet-tile, img.leaflet-tile')].map(function (img) {
        return img.getAttribute('src') || img.src || '';
      });
      const topo = tiles.some(function (s) { return /World_Topo_Map/i.test(s); });
      return {
        pass: terrainPressed || dataPressed || topo,
        terrainPressed,
        dataPressed,
        topo,
        sample: tiles.filter(Boolean)[0] || ''
      };
    });
    check(
      'default basemap is Terrain',
      terrain.pass,
      'aria=' + terrain.terrainPressed + ' data-style=' + terrain.dataPressed +
        ' topoTile=' + terrain.topo + ' sample=' + (terrain.sample || '').slice(0, 80)
    );

    // ---- 1. village marker tap stays open ----
    const villageName = targets.village && targets.village.name;
    let villageTapOk = false;
    if (!villageName) {
      check('village marker tap: popup stays 1500ms', false, 'no village items');
    } else {
      const revealed = await revealMarker(page, villageName);
      await sleep(500);
      if (!revealed.ok) {
        check('village marker tap: popup stays 1500ms', false,
          'could not uncluster ' + villageName + ' zoom=' + revealed.zoom);
      } else {
        await page.evaluate(() => { window.__popupLog = []; });
        const t0 = Date.now();
        const tap = await tapMarkerOrFallback(page, villageName);
        const early = await page.waitForSelector('.leaflet-popup', { timeout: 2500 }).catch(() => null);
        // Ghost map tap: leftover click after autoPan (phone dismiss).
        const ghost = await tapEmptyMap(page);
        await sleep(Math.max(400, 1500 - (Date.now() - t0)));
        const snap = await popupSnapshot(page);
        villageTapOk = !!(snap.present && /village/i.test(snap.type + ' ' + snap.name));
        const log = (snap.log || []).map((e) => e.ev).join(',');
        check(
          'village marker tap: popup stays 1500ms',
          villageTapOk,
          (snap.present ? 'open "' + snap.name + '"' : 'popup gone') +
            ' target=' + villageName + ' early=' + !!early +
            ' ghostMap=' + ghost.tapped + ' log=' + log +
            (tap.ok ? '' : ' tap=' + tap.reason)
        );
      }
    }

    // ---- 2. resort marker tap stays open (no regression) ----
    const resortName = targets.resort && targets.resort.name;
    if (!resortName) {
      check('resort marker tap: popup stays 1500ms', false, 'no resort items');
    } else {
      const revealed = await revealMarker(page, resortName);
      await sleep(500);
      if (!revealed.ok) {
        check('resort marker tap: popup stays 1500ms', false,
          'could not uncluster ' + resortName);
      } else {
        await page.evaluate(() => { window.__popupLog = []; });
        const tap = await tapMarkerOrFallback(page, resortName);
        const early = await page.waitForSelector('.leaflet-popup', { timeout: 2500 }).catch(() => null);
        await sleep(1500);
        const snap = await popupSnapshot(page);
        const ok = !!(snap.present && ( /resort|ski/i.test(snap.type) || snap.name === resortName ));
        check(
          'resort marker tap: popup stays 1500ms',
          ok,
          (snap.present ? 'open "' + snap.name + '"' : 'popup gone') +
            ' target=' + resortName + ' early=' + !!early +
            (tap.ok ? '' : ' tap=' + tap.reason)
        );
      }
    }

    // ---- 3. village popup inside visual viewport ----
    // Prefer the tap-opened village popup; if it dismissed, open via the marker API
    // so this remains a layout assertion (centering / autoPan), not a second copy of test 1.
    let viewportSnap = await popupSnapshot(page);
    const villagePopupUp = viewportSnap.present && /village/i.test(viewportSnap.type);
    if (!villagePopupUp && villageName) {
      await page.evaluate((name) => {
        const it = items.find(function (i) { return i.name === name; });
        if (!it) return;
        map.setView([it.lat, it.lng], Math.max(map.getZoom(), 13), { animate: false });
        it.marker.openPopup();
      }, villageName);
      await sleep(700);
      viewportSnap = await popupSnapshot(page);
    }
    check(
      'village popup stays inside visual viewport',
      !!(viewportSnap.present && viewportSnap.nearly),
      viewportSnap.present
        ? (viewportSnap.fully ? 'fully in view' : (viewportSnap.nearly ? 'nearly in view' : 'overflow ' + JSON.stringify(viewportSnap.overflow))) +
          ' ' + Math.round(viewportSnap.rect.width) + 'x' + Math.round(viewportSnap.rect.height)
        : 'no popup'
    );

    // ---- 9. village popup still has property listing links ----
    if (!viewportSnap.present || !/village/i.test(viewportSnap.type)) {
      if (villageName) {
        await page.evaluate((name) => {
          const it = items.find(function (i) { return i.name === name; });
          if (it) it.marker.openPopup();
        }, villageName);
        await sleep(400);
        viewportSnap = await popupSnapshot(page);
      }
    }
    check(
      'village popup contains a property listing link',
      !!(viewportSnap.present && viewportSnap.propertyLinks >= 1),
      viewportSnap.present
        ? viewportSnap.propertyLinks + ' .popup-link.property on "' + viewportSnap.name + '"'
        : 'no village popup'
    );

    // ---- 4. list-row tap for a unique village ----
    const uniqueName = targets.uniqueVillage;
    if (!uniqueName) {
      check('village list-row tap: popup opens and stays', false, 'no unique village name');
    } else {
      await page.evaluate(() => { if (map.closePopup) map.closePopup(); });
      const opened = await openDrawer(page);
      await page.fill('#search', uniqueName);
      await sleep(400);
      const rowInfo = await page.evaluate((name) => {
        const rows = [...document.querySelectorAll('#resultList .result')];
        const row = rows.find(function (r) { return r.textContent.indexOf(name) !== -1; });
        if (!row) return { n: rows.length };
        const r = row.getBoundingClientRect();
        return { n: rows.length, x: r.x + r.width / 2, y: r.y + Math.min(18, r.height / 2), w: r.width, h: r.height };
      }, uniqueName);
      if (!opened && !(rowInfo.w > 0)) {
        check('village list-row tap: popup opens and stays', false, 'drawer did not open');
      } else if (!rowInfo.w) {
        check('village list-row tap: popup opens and stays', false,
          'no .result row for ' + uniqueName + ' (rows=' + rowInfo.n + ')');
      } else {
        await page.evaluate(() => { window.__popupLog = []; });
        await realTap(page, rowInfo.x, rowInfo.y);
        const early = await page.waitForSelector('.leaflet-popup', { timeout: 5000 }).catch(() => null);
        await sleep(1500);
        const snap = await popupSnapshot(page);
        const nameMatch = snap.present && snap.name.indexOf(uniqueName) !== -1;
        check(
          'village list-row tap: popup opens and stays',
          !!(snap.present && nameMatch),
          (snap.present ? 'open "' + snap.name + '"' : 'popup gone') +
            ' row=' + uniqueName + ' early=' + !!early
        );
      }
    }

    // ---- 9b. shield must not swallow a deliberate tap on a different pin ----
    const second = await secondPinScenarios(page, 3);
    if (!second.scenarios.length) {
      check('tapping a second pin during the shield window opens its popup', false,
        'inconclusive - no fair two-pin scenario (centres=' + second.stats.centres +
          ' pairs=' + second.stats.pairs + ' exposed=' + second.stats.exposed +
          ' driftedOut=' + second.stats.driftedOut +
          ' tapMissed=' + second.stats.tapMissed +
          ' shieldExpired=' + second.stats.shieldExpired +
          ' lastBoxes=' + second.stats.lastBoxes + ' lastIcons=' + second.stats.lastIcons + ')');
    } else {
      check(
        'tapping a second pin during the shield window opens its popup',
        second.scenarios.every((sc) => sc.ok),
        second.scenarios.map((sc) => sc.a + '->' + sc.b + '=' + (sc.ok ? 'ok' : '"' + sc.got + '"')).join('; ') +
          ' (tap +' + SHIELD_TAP_DELAY_MS + 'ms)'
      );
    }

    // ---- 10. desktop still shows filter controls ----
    const desktop = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      hasTouch: false
    });
    const desk = await desktop.newPage();
    await desk.goto(origin + '/', { waitUntil: 'load', timeout: 60000 });
    await waitForMap(desk);
    const deskFilters = await desk.evaluate(() => {
      const el = document.querySelector('.controls');
      if (!el) return { pass: false, reason: '.controls missing' };
      const cs = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      const visible = cs.display !== 'none' && cs.visibility !== 'hidden' && r.height > 8 && r.width > 40;
      const country = document.querySelector('#countryGroup, .country-btn');
      const types = document.querySelector('#btnResorts, #btnVillages, .toggle-btn');
      const countryVisible = country && getComputedStyle(country).display !== 'none';
      const typesVisible = types && getComputedStyle(types).display !== 'none';
      return {
        pass: visible && !!(countryVisible || typesVisible),
        reason: 'display=' + cs.display + ' ' + Math.round(r.width) + 'x' + Math.round(r.height) +
          ' country=' + !!countryVisible + ' types=' + !!typesVisible
      };
    });
    check('desktop 1440x900 still shows filter controls', deskFilters.pass, deskFilters.reason);

    // ---- 11. desktop legend can be dismissed and stays dismissed ----
    const legend = await desk.evaluate(async () => {
      const el = document.getElementById('legend');
      const btn = document.getElementById('legendToggle');
      if (!el || !btn) return { pass: false, reason: 'legend or toggle missing' };
      const openH = el.getBoundingClientRect().height;
      btn.click();
      await new Promise((r) => setTimeout(r, 250));
      const closedH = el.getBoundingClientRect().height;
      let stored = null;
      try { stored = localStorage.getItem('alps.legend'); } catch (_) {}
      const bodyHidden = getComputedStyle(document.getElementById('legendBody')).display === 'none';
      btn.click();
      return {
        pass: bodyHidden && closedH < openH - 20 && stored === 'collapsed',
        reason: 'open=' + Math.round(openH) + 'px collapsed=' + Math.round(closedH) +
          'px bodyHidden=' + bodyHidden + ' stored=' + stored
      };
    });
    check('desktop legend collapses and the choice persists', legend.pass, legend.reason);

    // ---- 11b. app height tracks a viewport change (JS sizing is live) ----
    // The band only shows up on iOS, where dvh can outrun the visible area, so
    // assert the JS sizing is live rather than a static dvh. Runs on its own
    // page: resizing the shared mobile page would move the map under the
    // popup checks above.
    const probeCtx = await browser.newContext({
      viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, deviceScaleFactor: 2
    });
    const probe = await probeCtx.newPage();
    await probe.goto(origin + '/', { waitUntil: 'load', timeout: 60000 });
    await waitForMap(probe);
    const readAppH = () => probe.evaluate(() => {
      const vh = window.visualViewport ? window.visualViewport.height : window.innerHeight;
      const appH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--app-h'));
      return {
        appH: Number.isFinite(appH) ? appH : null,
        vh: Math.round(vh),
        bodyH: Math.round(document.body.getBoundingClientRect().height),
        stageBottom: Math.round(document.querySelector('.stage').getBoundingClientRect().bottom)
      };
    });
    const before = await readAppH();
    await probe.setViewportSize({ width: 390, height: 640 });
    await sleep(350);
    const after = await readAppH();
    await probeCtx.close();
    check(
      'app height tracks a viewport change (JS sizing is live)',
      before.appH === 844 && after.appH === 640 &&
        after.bodyH === 640 && after.stageBottom === 640,
      '844 -> --app-h=' + before.appH + '; 640 -> --app-h=' + after.appH +
        ' body=' + after.bodyH + ' stageBottom=' + after.stageBottom
    );

    // ---- 12. no JS errors on the mobile page ----
    check(
      'no JS/console errors on mobile load',
      jsErrors.length === 0,
      jsErrors.length ? jsErrors.slice(0, 3).join(' | ') : 'clean'
    );

    await desktop.close();
    await mobile.close();
  } catch (err) {
    check('harness', false, (err && err.stack) || String(err));
  } finally {
    if (browser) await browser.close().catch(() => {});
    await new Promise((r) => server.close(r));
    const report = writeReport(extra);
    console.log('\n' + report.score + ' passed');
    const failed = results.filter((r) => !r.pass);
    if (failed.length) {
      console.log('FAILED: ' + failed.map((f) => f.name).join('; '));
    }
    process.exit(results.length && results.every((r) => r.pass) ? 0 : 1);
  }
})();
