/* Dynamo landing page interactions */
(function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const fmt = (n) => Math.round(n).toLocaleString('en-US');
  const fmtTok = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : String(Math.round(n));

  /* ---------- Toast + clipboard ---------- */
  const toast = $('#toast');
  let toastTimer;
  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1800);
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); showToast('Copied to clipboard'); }
    catch (e) { showToast('Copy failed — select manually'); }
  }

  /* ---------- Header ---------- */
  const header = $('#site-header');
  window.addEventListener('scroll', () => header.classList.toggle('scrolled', window.scrollY > 8), { passive: true });
  const toggle = $('#nav-toggle');
  toggle.addEventListener('click', () => {
    const open = header.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  $$('#nav-links a').forEach(a => a.addEventListener('click', () => {
    header.classList.remove('open'); toggle.setAttribute('aria-expanded', 'false');
  }));

  /* ---------- Install copy ---------- */
  $('#install-cmd').addEventListener('click', () => copy('pip install agent-visual-harness'));

  /* ---------- Code tabs ---------- */
  const tabs = $$('.code-tab');
  tabs.forEach(tab => tab.addEventListener('click', () => {
    tabs.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
    tab.classList.add('active'); tab.setAttribute('aria-selected', 'true');
    $$('.code-panel').forEach(p => p.classList.toggle('active', p.dataset.panel === tab.dataset.tab));
  }));
  $('#code-copy').addEventListener('click', () => {
    const panel = $('.code-panel.active');
    if (panel) copy(panel.innerText);
  });

  /* ---------- Hero simulation ---------- */
  const screen = $('#hero-screen');
  const cursor = $('#hero-cursor');
  const ringLayer = $('#ring-layer');
  const telemetry = $('#telemetry');
  const kfTrack = $('#kf-track');
  const kfCount = $('#kf-count');
  const timerEl = $('#rec-timer');
  const btn = $('#fake-btn');
  const cards = $$('.fake-card');

  const start = performance.now();
  let keyframes = 0;
  const FPS = 30;

  function elapsed() { return (performance.now() - start) / 1000; }
  function tick() {
    const t = elapsed();
    const m = Math.floor(t / 60) % 60;
    const s = (t % 60).toFixed(1).padStart(4, '0');
    timerEl.textContent = String(m).padStart(2, '0') + ':' + s;
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  function logLine(action, x, y) {
    const t = elapsed().toFixed(2);
    const line = document.createElement('div');
    line.className = 'tel-line';
    let html = '{<span class="k">"timestamp"</span>: <span class="n">' + t + '</span>, <span class="k">"action"</span>: <span class="v">"' + action + '"</span>';
    if (x !== undefined) html += ', <span class="k">"x"</span>: <span class="n">' + x + '</span>, <span class="k">"y"</span>: <span class="n">' + y + '</span>';
    line.innerHTML = html + '}';
    telemetry.appendChild(line);
    while (telemetry.children.length > 5) telemetry.removeChild(telemetry.firstChild);
  }

  function addKeyframe(isAction) {
    keyframes++;
    const kf = document.createElement('div');
    kf.className = 'kf' + (isAction ? ' action' : '');
    kf.innerHTML = '<span>' + elapsed().toFixed(1) + 's</span>';
    kfTrack.appendChild(kf);
    while (kfTrack.children.length > 8) kfTrack.removeChild(kfTrack.firstChild);
    const total = Math.max(1, Math.round(elapsed() * FPS));
    kfCount.textContent = keyframes + ' / ' + fmt(total) + ' frames kept';
  }

  function ring(px, py) {
    const r = document.createElement('div');
    r.className = 'click-ring';
    r.style.left = px + '%'; r.style.top = py + '%';
    ringLayer.appendChild(r);
    setTimeout(() => r.remove(), 950);
  }

  // Scripted agent actions (percent coordinates in the screen)
  const script = [
    { x: 22, y: 50, act: 'click', fx: () => { cards.forEach(c => c.classList.remove('active')); cards[0].classList.add('active'); } },
    { x: 46, y: 50, act: 'click', fx: () => { cards.forEach(c => c.classList.remove('active')); cards[1].classList.add('active'); } },
    { x: 46, y: 50, act: 'scroll', noRing: true },
    { x: 20, y: 66, act: 'click', fx: () => { btn.classList.add('pressed'); setTimeout(() => btn.classList.remove('pressed'), 500); } },
    { x: 70, y: 50, act: 'click', fx: () => { cards.forEach(c => c.classList.remove('active')); cards[2].classList.add('active'); } },
    { x: 60, y: 82, act: 'type', noRing: true }
  ];
  let i = 0;
  function step() {
    const a = script[i % script.length];
    cursor.style.left = a.x + '%';
    cursor.style.top = a.y + '%';
    setTimeout(() => {
      const w = screen.clientWidth, h = screen.clientHeight;
      const px = Math.round(a.x / 100 * 1920), py = Math.round(a.y / 100 * 1080);
      if (!a.noRing) ring(a.x, a.y);
      if (a.fx) a.fx();
      logLine(a.act, a.noRing ? undefined : px, py);
      addKeyframe(true);
      // Occasionally emit a scene-change keyframe shortly after
      if (Math.random() > 0.45) setTimeout(() => addKeyframe(false), 450);
      void w; void h;
    }, 950);
    i++;
  }
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    setTimeout(step, 600);
    setInterval(step, 2200);
  } else {
    logLine('click', 420, 680); addKeyframe(true);
  }

  /* ---------- Token calculator ---------- */
  const inD = $('#in-duration'), inF = $('#in-fps'), inA = $('#in-actions'), inS = $('#in-scene'), inT = $('#in-tpi');
  const sceneLabels = ['', 'Static', 'Low', 'Moderate', 'High', 'Very high'];
  const scenePerMin = [0, 4, 8, 14, 24, 40]; // scene-change keyframes per minute
  function calc() {
    const d = +inD.value, f = +inF.value, a = +inA.value, s = +inS.value, tpi = +inT.value;
    $('#out-duration').textContent = d >= 60 ? (d / 60).toFixed(d % 60 ? 1 : 0) + ' min' : d + ' s';
    $('#out-fps').textContent = f + ' fps';
    $('#out-actions').textContent = a;
    $('#out-scene').textContent = sceneLabels[s];
    $('#out-tpi').textContent = '~' + fmt(tpi);

    const raw = d * f;
    // action frames + scene frames, with ~25% overlap deduplicated
    const scene = scenePerMin[s] * d / 60;
    const kept = Math.min(raw, Math.max(1, Math.round((a + scene) * 0.8 + 1)));
    const rawTok = raw * tpi, keptTok = kept * tpi;
    const reduction = raw ? (1 - kept / raw) * 100 : 0;

    $('#res-raw').textContent = fmt(raw);
    $('#res-kept').textContent = fmt(kept);
    $('#res-raw-tok').textContent = fmtTok(rawTok) + ' tokens';
    $('#res-kept-tok').textContent = fmtTok(keptTok) + ' tokens';
    $('#bar-kept').style.width = Math.max(0.5, kept / raw * 100) + '%';
    $('#res-reduction').textContent = reduction.toFixed(1) + '%';
  }
  [inD, inF, inA, inS, inT].forEach(el => el.addEventListener('input', calc));
  calc();

  /* ---------- Reveal on scroll ---------- */
  const revealTargets = $$('.section-head, .problem-card, .module-card, .artifact-card, .calc, .table-wrap, .enterprise, .faq-item, .demo-form, .code-window');
  revealTargets.forEach(el => el.classList.add('reveal'));
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.12 });
    revealTargets.forEach(el => io.observe(el));
  } else {
    revealTargets.forEach(el => el.classList.add('in'));
  }

  /* ---------- Demo form ---------- */
  const form = $('#demo-form');
  const status = $('#form-status');
  const submitBtn = $('#demo-submit');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    status.className = 'form-status'; status.textContent = '';
    let valid = true;
    $$('[required]', form).forEach(el => {
      const ok = el.value.trim() && (el.type !== 'email' || /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(el.value.trim()));
      el.classList.toggle('invalid', !ok);
      if (!ok) valid = false;
    });
    if (!valid) { status.classList.add('err'); status.textContent = 'Please complete the highlighted fields.'; return; }

    const data = Object.fromEntries(new FormData(form).entries());
    submitBtn.disabled = true;
    submitBtn.innerHTML = 'Sending… <i class="fa-solid fa-spinner fa-spin"></i>';
    try {
      const res = await fetch('tables/demo_requests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      form.reset();
      status.classList.add('ok');
      status.textContent = '✓ Thanks — an engineer will reach out within one business day.';
    } catch (err) {
      status.classList.add('err');
      status.textContent = 'Something went wrong. Please try again.';
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = 'Request a demo <i class="fa-solid fa-arrow-right"></i>';
    }
  });
  $$('input, select, textarea', form).forEach(el => el.addEventListener('input', () => el.classList.remove('invalid')));
})();
