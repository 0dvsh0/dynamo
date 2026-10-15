/* Paints a payment-gateway transaction log into a <canvas>.
 *
 * This is the demo's OCR target. The point of a canvas is that nothing about
 * it is readable without looking at the pixels: no textContent, no form value,
 * no accessibility node. Everything an agent could scrape out of the DOM is
 * deliberately absent, so an OCR benchmark run against this page is measuring
 * the real capability rather than a convenient one.
 */
(() => {
  "use strict";

  const LOG_LINES = [
    { level: "info", ts: "09:12:41.882Z", tag: "gateway.session", text: "client handshake ok proto=h2 cipher=TLS_AES_256_GCM" },
    { level: "info", ts: "09:12:42.104Z", tag: "gateway.route",   text: "txn 7f3c9ab1 -> acquirer NORTHWIND_SANDBOX pool=primary" },
    { level: "warn", ts: "09:12:42.660Z", tag: "gateway.risk",    text: "velocity check score=0.91 rule=velocity_exceeded" },
    { level: "warn", ts: "09:12:43.118Z", tag: "gateway.avs",     text: "AVS match=9/9 (exact) postal=97209 expected=97209" },
    { level: "info", ts: "09:12:43.402Z", tag: "gateway.auth",    text: "authorization request amount=184.50 currency=USD" },
    { level: "info", ts: "09:12:43.918Z", tag: "issuer.acquirer", text: "auth response received in 516ms network=eu-west-1" },
    { level: "error",ts: "09:12:44.201Z", tag: "issuer.acquirer", text: "DECLINED reason=do_not_honor response.code=05 response.text=Do not honor" },
    { level: "error",ts: "09:12:44.204Z", tag: "gateway.risk",    text: "decline.code=DO_NOT_HONOR issuer=CHASE card.brand=VISA last4=4242" },
    { level: "error",ts: "09:12:44.229Z", tag: "gateway.ledger",  text: "void auth_000188374 txn=7f3c9ab1 amount=0.00 refund=none" },
    { level: "fatal",ts: "09:12:44.310Z", tag: "gateway.txn",     text: "terminal state=FAILED retryable=false attempt=3 of 3" },
    { level: "info", ts: "09:12:44.312Z", tag: "gateway.session", text: "session 1f8c close reason=terminal_state http.status=402" },
  ];

  const COLOURS = {
    info:  { key: "#58a6ff", text: "#adbac7" },
    warn:  { key: "#d29922", text: "#d5b46a" },
    error: { key: "#f85149", text: "#ff9d96" },
    fatal: { key: "#ff7b72", text: "#ffdcd7", bg: "#2d1214" },
  };

  const FONT = "15px ui-monospace, 'DejaVu Sans Mono', Menlo, Consolas, monospace";
  const PAD = 16;
  const LINE_H = 38;

  function paint(canvas) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#010409";
    ctx.fillRect(0, 0, w, h);

    // A header rule, so a diff has a non-text edge to work with.
    ctx.fillStyle = "#21262d";
    ctx.fillRect(0, 0, w, 30);
    ctx.font = "600 13px ui-monospace, 'DejaVu Sans Mono', Menlo, Consolas, monospace";
    ctx.fillStyle = "#7d8590";
    ctx.fillText("NW-OUTFITTERS-4417 · sandbox · 2026-10-14T09:12:44Z", PAD, 20);

    ctx.font = FONT;
    ctx.textBaseline = "alphabetic";

    LOG_LINES.forEach((line, i) => {
      const y = 30 + PAD + i * LINE_H + 22;
      const palette = COLOURS[line.level] || COLOURS.info;

      if (palette.bg) {
        ctx.fillStyle = palette.bg;
        ctx.fillRect(0, y - 20, w, LINE_H);
      }
      if (line.level === "error" || line.level === "fatal") {
        ctx.fillStyle = palette.key;
        ctx.fillRect(0, y - 20, 3, LINE_H);
      }

      let x = PAD;
      ctx.fillStyle = "#484f58";
      x = draw(ctx, line.ts, x, y);
      x = draw(ctx, line.tag, x + 18, y, palette.key);
      x = draw(ctx, line.text, x + 200, y, palette.text);
      void x;
    });
  }

  function draw(ctx, text, x, y, colour) {
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);
    return x + ctx.measureText(text).width;
  }

  const canvas = document.getElementById("log");
  if (canvas) {
    paint(canvas);
    window.__gatewayLog = LOG_LINES;   // ground truth for the benchmark to score against
    window.__gatewayPainted = true;
  }

  // Proof for the demo page that the DOM really does hold none of this text.
  const probe = document.getElementById("probe");
  if (probe) {
    probe.addEventListener("click", () => {
      const el = document.getElementById("log");
      const out = document.getElementById("probe-out");
      const text = (el.textContent || "").trim();
      out.textContent = text
        ? `canvas.textContent = ${JSON.stringify(text.slice(0, 60))}`
        : 'canvas.textContent = "" — nothing to scrape; pixels only';
    });
  }

  // Real UI behaviour: retrying re-renders with a fresh attempt.
  const retry = document.getElementById("retry");
  if (retry) {
    retry.addEventListener("click", () => {
      LOG_LINES.push({
        level: "info",
        ts: "09:13:02.410Z",
        tag: "gateway.route",
        text: "manual retry queued operator=guest txn=7f3c9ab1 attempt=4",
      });
      paint(canvas);
      window.__gatewayLog = LOG_LINES;
    });
  }
})();