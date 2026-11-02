/* Landing page: theme toggle, owner-editable copy from /api/v1/landing, service version from
   /health, and the trial-request form. Nothing here decides anything about answers. */
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function applyTheme(t) { document.documentElement.dataset.theme = t; $("theme").textContent = t === "light" ? "☾" : "☀"; }
  (function initTheme() { let t = null; try { t = localStorage.getItem("aeos.theme"); } catch (e) {} if (!t) t = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark"; applyTheme(t); })();
  $("theme").addEventListener("click", () => { const t = document.documentElement.dataset.theme === "light" ? "dark" : "light"; applyTheme(t); try { localStorage.setItem("aeos.theme", t); } catch (e) {} });
  function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => { t.hidden = true; }, 4000); }

  fetch("/api/v1/landing").then((r) => r.json()).then((c) => {
    if (c.tagline) $("tagline").textContent = c.tagline;
    if (c.subtitle) $("subtitle").textContent = c.subtitle;
    $("foot-org").textContent = c.organisation || "AEOS Medical";
    if (c.github_url) $("foot-github").href = c.github_url;
    $("foot-contact").textContent = c.contact_email ? c.contact_email : "contact via the form above";
    $("shape-cards").innerHTML = (c.shapes || []).map((s) => `<div class="card"><h3>${esc(s.name)}</h3><div class="for">${esc(s.for)}</div><p>${esc(s.includes)}</p><div class="price">${esc(s.price)}</div></div>`).join("");
    if (!c.demo_enabled) { document.querySelectorAll("#cta-demo,#cta-demo-top").forEach((a) => { a.href = "/app"; a.textContent = "Open the app"; }); $("demo-note").textContent = "Self-service demo is off on this server."; }
    else $("demo-note").textContent = `The demo runs on a synthetic corpus with ${c.demo_questions} questions per visitor.`;
  }).catch(() => {});
  fetch("/health").then((r) => r.json()).then((h) => { $("foot-version").textContent = `${h.profile} · ${h.bundle_id} · gate ${h.gate_calibrated_on}`; }).catch(() => { $("foot-version").textContent = "offline"; });

  document.querySelectorAll(".shots img").forEach((img) => { img.addEventListener("error", () => { img.closest("figure").style.display = "none"; }); });

  $("lead-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target; const body = Object.fromEntries(new FormData(f).entries());
    $("lead-submit").disabled = true; $("lead-status").textContent = "sending…";
    try {
      const r = await fetch("/api/v1/leads", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
      $("lead-status").textContent = "Thank you — we will get back to you."; f.reset(); toast("Request received.");
    } catch (e) { $("lead-status").textContent = e.message; $("lead-submit").disabled = false; }
  });
})();
