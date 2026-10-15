/* Northwind Outfitters — demo storefront.
 *
 * A deliberately ordinary multi-step checkout: real form validation, an async
 * payment step, a decline, a retry, and a confirmation. Nothing here is faked for
 * the demo camera — it is the kind of flow a browser-automation agent actually
 * gets pointed at in CI.
 */
(function () {
  "use strict";

  const state = {
    step: "details",
    cart: [
      { sku: "NW-4471", name: "Ridgeline Hardshell Jacket", variant: "Slate / M", price: 18900, thumb: "NW" },
      { sku: "NW-2208", name: "Trailhead Merino Base Layer", variant: "Heather / L", price: 6400, thumb: "NW" },
      { sku: "NW-9915", name: "Summit 30L Trekking Pack", variant: "Moss", price: 14200, thumb: "NW" },
    ],
    details: { email: "demo@northwind.test", name: "", line1: "", city: "", zip: "", country: "US" },
    card: { number: "", name: "", exp: "", cvc: "" },
    attempts: 0,
    declined: false,
    order: null,
  };

  const view = document.getElementById("view");
  const stepsEl = document.getElementById("steps");
  const cartCount = document.getElementById("cart-count");

  const money = (cents) => "$" + (cents / 100).toFixed(2);
  const subtotal = () => state.cart.reduce((n, item) => n + item.price, 0);
  const shipping = () => (subtotal() > 25000 ? 0 : 795);
  const tax = () => Math.round((subtotal() + shipping()) * 0.0875);
  const total = () => subtotal() + shipping() + tax();

  function el(html) {
    // Templates here have several top-level siblings (heading, lede, columns), so
    // wrap them: a bare template's firstElementChild would silently keep only the
    // <h1> and drop everything after it.
    const t = document.createElement("template");
    t.innerHTML = `<div>${html.trim()}</div>`;
    return t.content.firstElementChild;
  }

  function layout(html) {
    // The order summary is a node, not markup, so it is dropped into a slot
    // after parsing rather than interpolated (which would stringify it).
    const node = el(html);
    const slot = node.querySelector("#summary-slot");
    if (slot) slot.replaceWith(summaryCard());
    return node;
  }

  function setStep(step) {
    state.step = step;
    const order = ["cart", "details", "payment", "review", "done"];
    const current = order.indexOf(step);
    stepsEl.querySelectorAll("li").forEach((li) => {
      const i = order.indexOf(li.dataset.step);
      li.className = i < current ? "done" : i === current ? "current" : "";
    });
    render();
  }

  function render() {
    cartCount.textContent = String(state.cart.length);
    if (state.step === "details") return renderDetails();
    if (state.step === "payment") return renderPayment();
    if (state.step === "review") return renderReview();
    return renderDone();
  }

  /* ------------------------------------------------------------- details */

  function renderDetails() {
    view.replaceChildren(layout(`
      <h1>Checkout</h1>
      <p class="lede">Step 2 of 4 — shipping and contact details.</p>
      <div class="cols">
        <div>
          <div class="card">
            <h2>Contact</h2>
            <div class="field">
              <label for="email">Email</label>
              <input id="email" type="email" value="${state.details.email}">
              <div class="err">Enter a valid email address.</div>
            </div>
          </div>
          <div class="card">
            <h2>Shipping address</h2>
            <div class="field">
              <label for="name">Full name</label>
              <input id="name" type="text" placeholder="Ada Lovelace" value="${state.details.name}">
              <div class="err">Full name is required.</div>
            </div>
            <div class="field">
              <label for="line1">Street address</label>
              <input id="line1" type="text" placeholder="18 Wharf Road" value="${state.details.line1}">
              <div class="err">Street address is required.</div>
            </div>
            <div class="row2">
              <div class="field">
                <label for="city">City</label>
                <input id="city" type="text" placeholder="Portland" value="${state.details.city}">
                <div class="err">City is required.</div>
              </div>
              <div class="field">
                <label for="zip">Postal code</label>
                <input id="zip" type="text" placeholder="97209" value="${state.details.zip}">
                <div class="err">Enter a 5-digit postal code.</div>
              </div>
            </div>
            <div class="field">
              <label for="country">Country</label>
              <select id="country">
                <option value="US"${state.details.country === "US" ? " selected" : ""}>United States</option>
                <option value="CA">Canada</option>
                <option value="GB">United Kingdom</option>
                <option value="DE">Germany</option>
              </select>
            </div>
            <div class="actions">
              <button class="btn-primary" id="to-payment">Continue to payment</button>
              <button class="btn-ghost" id="to-cart">Back to cart</button>
            </div>
          </div>
        </div>
        <div id="summary-slot"></div>
      </div>`));

    view.querySelector("#to-payment").addEventListener("click", submitDetails);
    view.querySelector("#to-cart").addEventListener("click", () => setStep("cart"));
    ["email", "name", "line1", "city", "zip"].forEach((id) => {
      const node = view.querySelector("#" + id);
      node.addEventListener("input", () => node.closest(".field").classList.remove("invalid"));
    });
  }

  function submitDetails() {
    const values = {};
    let firstBad = null;
    const ids = ["email", "name", "line1", "city", "zip", "country"];
    // Read every value first. Validating inside the same loop would test
    // values.country while it is still undefined, and the postal rule would
    // quietly never run.
    ids.forEach((id) => { values[id] = view.querySelector("#" + id).value.trim(); });

    ids.forEach((id) => {
      const node = view.querySelector("#" + id);
      let ok = values[id].length > 0;
      if (id === "email") ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values[id]);
      if (id === "zip" && values.country === "US") ok = /^\d{5}$/.test(values[id]);
      node.closest(".field").classList.toggle("invalid", !ok);
      if (!ok && !firstBad) firstBad = node;
    });
    if (firstBad) { firstBad.focus(); return; }
    Object.assign(state.details, values);
    setStep("payment");
  }

  /* ------------------------------------------------------------- payment */

  function renderPayment() {
    const declined = state.declined;
    view.replaceChildren(layout(`
      <h1>Payment</h1>
      <p class="lede">Step 3 of 4 — authorise ${money(total())}.</p>
      ${declined ? `<div class="banner err" id="declined">
        <span class="ic">✕</span>
        <div><b>Card declined by issuer</b>
        <span>Attempt ${state.attempts} was refused with <code>do_not_honor</code>. Check the card number or retry.</span></div>
      </div>` : ""}
      ${state.attempts > 1 && !declined ? `<div class="banner warn">
        <span class="ic">!</span>
        <div><b>Attempt ${state.attempts}</b><span>The previous decline has been logged against this order.</span></div>
      </div>` : ""}
      <div class="cols">
        <div>
          <div class="card">
            <h2>Card</h2>
            <div class="card-visual" id="card-visual">
              <div class="card-brand" id="cv-brand">NORTHWIND PAY</div>
              <div class="num" id="cv-num">•••• •••• •••• ••••</div>
              <div class="foot"><span id="cv-name">YOUR NAME</span><span id="cv-exp">MM/YY</span></div>
            </div>
            <div style="height:18px"></div>
            <div class="field">
              <label for="cnum">Card number</label>
              <input id="cnum" type="text" inputmode="numeric" placeholder="4242 4242 4242 4242" value="${state.card.number}">
              <div class="err">Enter a 16-digit card number.</div>
            </div>
            <div class="field">
              <label for="cname">Name on card</label>
              <input id="cname" type="text" placeholder="A. LOVELACE" value="${state.card.name}">
              <div class="err">Name on card is required.</div>
            </div>
            <div class="row2">
              <div class="field">
                <label for="cexp">Expiry</label>
                <input id="cexp" type="text" placeholder="04/29" value="${state.card.exp}">
                <div class="err">Use MM/YY.</div>
              </div>
              <div class="field">
                <label for="ccvc">CVC</label>
                <input id="ccvc" type="number" placeholder="123" value="${state.card.cvc}">
                <div class="err">3 digits.</div>
              </div>
            </div>
            <div class="pay-state" id="paying"><span class="spinner"></span>
              <span id="paying-text">Contacting issuer…</span></div>
            <div class="pay-result" id="pay-result"></div>
            <div class="actions">
              <button class="btn-primary" id="pay-btn">${declined ? "Retry payment" : "Pay " + money(total())}</button>
              <button class="btn-ghost" id="to-details">Back to details</button>
            </div>
          </div>
        </div>
        <div id="summary-slot"></div>
      </div>`));

    const num = view.querySelector("#cnum");
    const name = view.querySelector("#cname");
    const exp = view.querySelector("#cexp");
    num.addEventListener("input", () => {
      const digits = num.value.replace(/\D/g, "").slice(0, 16);
      num.value = digits.replace(/(.{4})/g, "$1 ").trim();
      const shown = digits.padEnd(16, "•");
      view.querySelector("#cv-num").textContent = shown.replace(/(.{4})/g, "$1 ").trim();
      view.querySelector("#cv-brand").textContent = digits.startsWith("4") ? "VISA" : "CARD";
      num.closest(".field").classList.remove("invalid");
    });
    name.addEventListener("input", () => {
      view.querySelector("#cv-name").textContent = (name.value || "YOUR NAME").toUpperCase();
      name.closest(".field").classList.remove("invalid");
    });
    exp.addEventListener("input", () => {
      let v = exp.value.replace(/\D/g, "").slice(0, 4);
      if (v.length >= 3) v = v.slice(0, 2) + "/" + v.slice(2);
      exp.value = v;
      view.querySelector("#cv-exp").textContent = v || "MM/YY";
      exp.closest(".field").classList.remove("invalid");
    });

    view.querySelector("#pay-btn").addEventListener("click", pay);
    view.querySelector("#to-details").addEventListener("click", () => setStep("details"));
  }

  function pay() {
    const values = {
      number: view.querySelector("#cnum").value.replace(/\D/g, ""),
      name: view.querySelector("#cname").value.trim(),
      exp: view.querySelector("#cexp").value.trim(),
      cvc: view.querySelector("#ccvc").value.trim(),
    };
    let firstBad = null;
    const checks = [
      [values.number.length === 16, "cnum"],
      [values.name.length > 0, "cname"],
      [/^\d{2}\/\d{2}$/.test(values.exp), "cexp"],
      [/^\d{3,4}$/.test(values.cvc), "ccvc"],
    ];
    checks.forEach(([ok, id]) => {
      const node = view.querySelector("#" + id);
      node.closest(".field").classList.toggle("invalid", !ok);
      if (!ok && !firstBad) firstBad = node;
    });
    if (firstBad) { firstBad.focus(); return; }
    Object.assign(state.card, values);

    state.attempts += 1;
    const btn = view.querySelector("#pay-btn");
    const spinner = view.querySelector("#paying");
    btn.disabled = true;
    spinner.style.display = "flex";
    view.querySelector("#paying-text").textContent = "Contacting issuer…";

    // The first two attempts are refused; the third goes through. An agent that
    // cannot see the screen will not notice that it needs to change something.
    setTimeout(() => {
      view.querySelector("#paying-text").textContent = "3-D Secure challenge…";
    }, 1100);

    setTimeout(() => {
      spinner.style.display = "none";
      btn.disabled = false;
      if (state.attempts <= 2) {
        state.declined = true;
        render();
        return;
      }
      state.declined = false;
      state.order = {
        id: "NW-" + (418260 + state.attempts * 7),
        placed: new Date().toISOString().replace("T", " ").slice(0, 19),
      };
      setStep("review");
    }, 2600);
  }

  /* -------------------------------------------------------------- review */

  function renderReview() {
    const o = state.order;
    view.replaceChildren(layout(`
      <h1>Review order</h1>
      <p class="lede">Step 4 of 4 — confirm and place the order.</p>
      <div class="cols">
        <div>
          <div class="card">
            <h2>Payment authorised</h2>
            <div class="banner ok"><span class="ic">✓</span>
              <div><b>Charge captured</b>
              <span>${money(total())} authorised on ${mask(state.card.number)} after ${state.attempts} attempt${state.attempts > 1 ? "s" : ""}.</span></div>
            </div>
            <div class="row2">
              <div>
                <div class="badge ok">ship to</div>
                <p style="margin:8px 0 0;font-size:14px">
                  ${state.details.name}<br>${state.details.line1}<br>
                  ${state.details.city} ${state.details.zip}, ${state.details.country}
                </p>
              </div>
              <div>
                <div class="badge warn">card</div>
                <p style="margin:8px 0 0;font-size:14px;font-family:var(--mono)">
                  ${mask(state.card.number)}<br>${state.card.name.toUpperCase()}<br>exp ${state.card.exp}
                </p>
              </div>
            </div>
          </div>
          <div class="card">
            <h2>Confirm</h2>
            <div class="actions">
              <button class="btn-primary" id="place">Place order · ${money(total())}</button>
              <button class="btn-ghost" id="back-payment">Back to payment</button>
            </div>
          </div>
        </div>
        <div id="summary-slot"></div>
      </div>`));

    view.querySelector("#place").addEventListener("click", () => {
      view.querySelector("#place").disabled = true;
      view.querySelector("#place").textContent = "Placing order…";
      setTimeout(() => setStep("done"), 1400);
    });
    view.querySelector("#back-payment").addEventListener("click", () => setStep("payment"));
  }

  /* ---------------------------------------------------------------- done */

  function renderDone() {
    const o = state.order;
    view.replaceChildren(layout(`
      <h1>Order confirmed</h1>
      <p class="lede">Thanks, ${state.details.name.split(" ")[0]} — a receipt is on its way.</p>
      <div class="cols">
        <div>
          <div class="card">
            <h2>Receipt</h2>
            <div class="banner ok" style="margin-bottom:16px"><span class="ic">✓</span>
              <div><b>Payment captured · ${money(total())}</b>
              <span>Authorised after ${state.attempts} attempt${state.attempts > 1 ? "s" : ""}. Confirmation sent to ${state.details.email}.</span></div>
            </div>
            <div class="receipt">order     ${o.id}
placed   ${o.placed}
email    ${state.details.email}
card     ${mask(state.card.number)}
subtotal ${money(subtotal())}
shipping ${shipping() === 0 ? "free" : money(shipping())}
tax      ${money(tax())}
total    ${money(total())}</div>
          </div>
          <div class="card">
            <div class="actions">
              <button class="btn-ghost" id="again">Start another order</button>
            </div>
          </div>
        </div>
        <div id="summary-slot"></div>
      </div>`));
    view.querySelector("#again").addEventListener("click", () => location.reload());
  }

  function mask(digits) {
    return "•••• •••• •••• " + digits.slice(-4);
  }

  function summaryCard() {
    return el(`
      <div class="card">
        <h2>Order summary</h2>
        ${state.cart.map((i) => `
          <div class="item">
            <div class="thumb">${i.thumb}</div>
            <div><div class="name">${i.name}</div><div class="meta">${i.variant} · ${i.sku}</div></div>
            <div class="price">${money(i.price)}</div>
          </div>`).join("")}
        <div class="totals" style="margin-top:18px">
          <div><span>Subtotal</span><span class="mono">${money(subtotal())}</span></div>
          <div><span>Shipping</span><span class="mono">${shipping() === 0 ? "Free" : money(shipping())}</span></div>
          <div><span>Estimated tax</span><span class="mono">${money(tax())}</span></div>
          <div class="grand"><span>Total</span><span class="mono">${money(total())}</span></div>
        </div>
      </div>`);
  }

  // Expose a tiny surface so an automation driver can reset between runs.
  window.__nw = {
    reset() { location.reload(); },
    state() { return { step: state.step, attempts: state.attempts, declined: state.declined }; },
  };

  setStep("details");
})();
