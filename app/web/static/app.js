/* AEOS Medical client. Renders what the API returns; nothing here decides anything — every action,
   label and number comes from the service (see docs/UI_PROTOTYPE_REVIEW.md). */
window.AEOS = (function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const state = { token: null, apiKey: null, principal: null, chatId: null, route: "chat", scope: { uploaded: true, indexed: true } };
  try { state.token = sessionStorage.getItem("aeos.token"); state.apiKey = sessionStorage.getItem("aeos.apikey"); } catch (e) {}

  function headers(json) {
    const h = json ? { "Content-Type": "application/json" } : {};
    if (state.token) h["Authorization"] = "Bearer " + state.token;
    else if (state.apiKey) h["X-API-Key"] = state.apiKey;
    return h;
  }
  async function api(method, path, body, raw) {
    const r = await fetch(path, { method, headers: headers(!raw && body !== undefined), body: raw ? body : (body !== undefined ? JSON.stringify(body) : undefined) });
    let data = null;
    try { data = await r.json(); } catch (e) { data = null; }
    if (!r.ok) { const err = new Error((data && data.error) || `HTTP ${r.status}`); err.status = r.status; err.data = data; throw err; }
    return data;
  }
  function toast(msg, ms) {
    const t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(t._h); t._h = setTimeout(() => { t.hidden = true; }, ms || 3500);
  }
  function decodeJwt(tok) {
    try { return JSON.parse(atob(tok.split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))); } catch (e) { return null; }
  }
  function updateWho() {
    const p = state.token ? decodeJwt(state.token) : null;
    state.principal = p ? { sub: p.sub, roles: p.roles || [], courses: p.courses || [] } : (state.apiKey ? { sub: "api-key", roles: [] } : null);
    $("who").textContent = state.principal ? `${state.principal.sub}${state.principal.roles.length ? " · " + state.principal.roles.join(", ") : ""}` : "anonymous";
    $("signin").textContent = state.principal ? "Sign out" : "Sign in";
    document.querySelectorAll(".nav a.admin").forEach((a) => { a.style.display = (state.principal && state.principal.roles.includes("admin")) ? "" : "none"; });
    loadRecent();
  }
  function signin() {
    if (state.principal) { state.token = null; state.apiKey = null; try { sessionStorage.removeItem("aeos.token"); sessionStorage.removeItem("aeos.apikey"); } catch (e) {} state.chatId = null; updateWho(); route(); return; }
    const v = prompt("Paste an access token (JWT) or an API key (lbr_…)");
    if (!v) return;
    if (v.startsWith("lbr_")) { state.apiKey = v.trim(); try { sessionStorage.setItem("aeos.apikey", state.apiKey); } catch (e) {} }
    else { state.token = v.trim(); try { sessionStorage.setItem("aeos.token", state.token); } catch (e) {} }
    updateWho(); route();
  }
  async function loadRecent() {
    const el = $("recent");
    if (!state.principal) { el.innerHTML = '<span class="muted small" style="padding:0 12px">sign in to keep chats</span>'; return; }
    try {
      const chats = await api("GET", "/api/v1/chats");
      el.innerHTML = chats.length ? chats.map((c) => `<a href="#/chat/${esc(c.chat_id)}" title="${esc(c.title)}">${esc(c.title)}</a>`).join("") : '<span class="muted small" style="padding:0 12px">no chats yet</span>';
    } catch (e) { el.innerHTML = `<span class="muted small" style="padding:0 12px">${esc(e.message)}</span>`; }
  }

  // ---- router -----------------------------------------------------------------------------
  const pages = {};
  function register(name, fn) { pages[name] = fn; }
  async function route() {
    const hash = location.hash.replace(/^#\/?/, "") || "chat";
    const [name, ...rest] = hash.split("/");
    state.route = name;
    document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === name));
    const main = $("main"); main.className = "main" + (name === "chat" ? " chatlayout" : "");
    const fn = pages[name] || pages.chat;
    main.innerHTML = '<div class="card muted">loading…</div>';
    try { await fn(main, rest); } catch (e) { main.innerHTML = `<div class="card err">${esc(e.message)}</div>`; }
  }
  window.addEventListener("hashchange", route);
  $("signin").addEventListener("click", signin);
  function applyTheme(t) { document.documentElement.dataset.theme = t; $("theme").textContent = t === "light" ? "☾" : "☀"; $("theme").title = t === "light" ? "switch to dark theme" : "switch to light theme"; }
  (function initTheme() { let t = null; try { t = localStorage.getItem("aeos.theme"); } catch (e) {} if (!t) t = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark"; applyTheme(t); })();
  $("theme").addEventListener("click", () => { const t = document.documentElement.dataset.theme === "light" ? "dark" : "light"; applyTheme(t); try { localStorage.setItem("aeos.theme", t); } catch (e) {} });
  $("topsearch").addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.value.trim()) { location.hash = "#/chat"; setTimeout(() => { const q = $("q"); if (q) { q.value = e.target.value; $("ask").requestSubmit(); } }, 50); } });
  document.addEventListener("DOMContentLoaded", () => { updateWho(); route(); });

  // ---- shared renderers -------------------------------------------------------------------
  const ACTION_TEXT = {
    ANSWER: "Answer supported by the cited evidence",
    ANSWER_WITH_CAUTION: "Delivered with caution: some claims were removed or sources disagree",
    CLARIFY: "The question needs clarification before it can be answered from the evidence",
    ABSTAIN: "No answer: the evidence does not support a reliable answer",
    ESCALATE: "Out of scope for this tool",
  };
  const chip = (t, c) => `<span class="chip ${c || ""}">${esc(t)}</span>`;
  const labelCls = (l) => (l === "supported" ? "ok" : l === "contradicted" ? "bad" : "warn");
  const tierCls = (t) => (t === "peer_reviewed" ? "ok" : t === "course_material" ? "acc" : "warn");
  function fmtPct(x) { return x == null ? "n/a" : (100 * x).toFixed(0) + "%"; }

  function renderBanner(b) {
    const conf = b.confidence == null ? "n/a" : (b.confidence_calibrated ? b.confidence.toFixed(2) : `${b.confidence.toFixed(2)} (uncalibrated)`);
    const review = b.review ? `<div class="small" style="margin-top:6px">${chip("under human review", "warn")} ${esc(b.review.message)} <a href="#/review/mine">My cases</a></div>` : "";
    return `<div class="banner ${esc(b.action)}"><b>${esc(b.action.replace(/_/g, " "))}</b> — ${esc(ACTION_TEXT[b.action] || "")}
      <div class="small muted">Deciding rule ${esc(b.rule)}: ${esc(b.reason)} · confidence ${esc(conf)} · evidence sufficiency ${b.sufficiency == null ? "n/a" : b.sufficiency.toFixed(2)} · scope ${esc(b.scope)}</div>${review}</div>`;
  }
  function citeIndex(b) { const idx = {}; b.sources.forEach((s, i) => { if (!(s.citation_id in idx)) idx[s.citation_id] = i + 1; }); return idx; }
  function withCites(text, b) {
    const idx = citeIndex(b);
    return esc(text).replace(/\[((?:PMID|DOI|SLIDE|FILE):[^\]]+)\]/g, (m, id) => idx[id] ? `<span class="cite" data-cite="${esc(id)}">[${idx[id]}]</span>` : esc(m));
  }
  function renderAnswer(b) {
    const subs = b.claims.filter((c) => c.substitution);
    const e = b.entities;
    const keyPoints = b.claims.filter((c) => !c.removed && c.label === "supported");
    return `<div class="bubble">${b.interpreted_as ? `<div class="small muted">Interpreted as: <i>${esc(b.interpreted_as)}</i></div>` : ""}${renderBanner(b)}<p>${withCites(b.answer, b)}</p>
      ${keyPoints.length ? `<b>Key points</b><ul>${keyPoints.map((c) => `<li>${withCites(c.text, b)} ${c.citations.map((i) => `<span class="cite" data-cite="${esc(i)}">[${citeIndex(b)[i] || "?"}]</span>`).join("")}</li>`).join("")}</ul>` : ""}
      ${b.removed_claims.length ? `<details><summary class="muted">${b.removed_claims.length} claim(s) removed from this answer</summary><ul>${b.removed_claims.map((c) => `<li class="muted">${esc(c)}</li>`).join("")}</ul></details>` : ""}
      ${b.conflicts.length ? `<div class="banner ANSWER_WITH_CAUTION"><b>${b.conflicts.some((c) => c.kind === "lecture_vs_literature") ? "Lecture vs literature" : "Evidence is mixed"}</b>${b.conflicts.map((c) => `<div class="small">${c.kind === "lecture_vs_literature" ? "lecture says" : "supports"}: ${c.supporting_ids.map((i) => chip(i, "acc")).join("")} ${c.kind === "lecture_vs_literature" ? "literature says otherwise" : "contradicts"}: ${c.contradicting_ids.map((i) => chip(i, "bad")).join("")} <span class="muted">${esc(c.description)}</span></div>`).join("")}</div>` : ""}
      ${e.asked.length ? `<div class="small muted">Asked about: ${e.asked.map((x) => chip(x, e.missing_in_evidence.includes(x) ? "bad" : "ok")).join("")}${e.missing_in_evidence.length ? ` · not found in the evidence: ${e.missing_in_evidence.map(esc).join(", ")}` : ""}</div>` : ""}
      ${subs.length ? `<div class="err small">Entity substitution detected: the answer named <b>${esc(subs[0].substitution.claim_entity)}</b> where the question asked about <b>${esc(subs[0].substitution.asked_entity)}</b> (${esc(subs[0].substitution.reason || "near neighbour")}).</div>` : ""}
      <footer class="meta">${esc(b.disclosure)}<br>profile <b>${esc(b.versions.profile)}</b> · bundle ${esc(b.versions.bundle_id)} · prompt ${esc(b.versions.prompt_id)} · gate calibration ${esc(b.versions.gate_calibrated_on)} · request ${esc(b.request_id)} · ${b.timings_ms.total} ms${b.usage ? ` · ${b.usage.prompt_tokens} in / ${b.usage.completion_tokens} out tokens${b.usage.exact ? "" : " (est.)"}` : ""}</footer></div>`;
  }
  function renderClaims(b) {
    const byPid = Object.fromEntries(b.sources.map((s) => [s.passage_id, s]));
    if (!b.claims.length) return '<p class="muted">No claims were generated.</p>';
    return b.claims.map((c) => {
      const span = c.evidence_span && byPid[c.evidence_span.passage_id];
      const cspan = c.contradicting_span && byPid[c.contradicting_span.passage_id];
      return `<div class="claim ${c.removed ? "removed" : ""}">${chip(c.label, labelCls(c.label))}${c.critical ? chip("critical") : ""}${c.removed ? chip("removed", "warn") : ""}${c.off_target ? chip("off-target", "warn") : ""}${c.fabricated_ids.length ? chip("fabricated citation", "bad") : ""}${c.numeric_mismatches.length ? chip("number mismatch", "bad") : ""}${c.entity_missing.length ? chip("entity not in evidence", "bad") : ""}
        <div class="txt">${esc(c.text)}</div>
        <div class="small muted">cites ${c.citations.map((i) => chip(i, c.supporting_ids.includes(i) ? "ok" : (c.fabricated_ids.includes(i) ? "bad" : ""))).join("") || "nothing"} · support ${c.support.toFixed(2)} · contradiction ${c.contradiction.toFixed(2)}</div>
        ${span ? `<blockquote>${esc(span.text)}<br><span class="mono">${esc(span.citation_id)} · ${esc(span.display_source)} · chars ${span.char_start}–${span.char_end}</span></blockquote>` : ""}
        ${cspan ? `<blockquote class="err">contradicting: ${esc(cspan.text)}<br><span class="mono">${esc(cspan.citation_id)}</span></blockquote>` : ""}
        <div class="small">${chip("Is this claim right?")}<button class="btn ghost small fb" data-claim="${esc(c.id)}" data-label="correct">yes</button> <button class="btn ghost small fb" data-claim="${esc(c.id)}" data-label="incorrect">no</button></div></div>`;
    }).join("");
  }
  function renderSources(b, compact) {
    if (!b.sources.length) return '<p class="muted">No sources.</p>';
    const max = Math.max(...b.sources.map((s) => s.retrieval_score), 1e-9);
    return b.sources.map((s, i) => `<div class="src" id="src-${esc(s.citation_id)}"><div class="n">[${i + 1}]</div><div>
      <div class="t">${esc(s.display_source)}</div>
      <div class="small">${chip(s.tier.replace("_", " "), tierCls(s.tier))}${s.year ? chip(s.year) : ""}${s.journal ? chip(s.journal) : ""}<span class="mono">${esc(s.citation_id)}</span>${s.url ? ` <a href="${esc(s.url)}" target="_blank" rel="noopener">open</a>` : ""}</div>
      <div class="small muted">retrieval relevance ${s.retrieval_score.toFixed(3)}</div><div class="bar"><i style="width:${Math.max(4, 100 * s.retrieval_score / max)}%"></i></div>
      ${compact ? "" : `<div class="small muted" style="margin-top:4px">${esc(s.text)}</div>`}</div></div>`).join("");
  }
  function renderAnalysis(b) {
    const v = b.verification;
    const rows = [["Claims supported", v.entailment_rate], ["Citations resolved", v.citation_support_rate], ["Contradiction", v.contradiction_rate], ["Unsupported", v.unsupported_rate], ["Entity grounding", v.query_entity_grounding], ["Evidence sufficiency", b.sufficiency], ["Confidence" + (b.confidence_calibrated ? "" : " (uncalibrated)"), b.confidence]];
    return `<p class="small muted">Verification of this answer. These are checks against the cited passages, not accuracy measurements; measured performance is on the Evaluations page.</p>
      <div class="meter">${rows.map(([k, x]) => `<div>${esc(k)}<div class="bar"><i style="width:${x == null ? 0 : Math.round(100 * x)}%"></i></div></div><div class="mono">${fmtPct(x)}</div>`).join("")}</div>
      <details style="margin-top:8px"><summary class="muted small">decision trace</summary><pre class="log">${esc(b.trace.join("\n"))}</pre></details>`;
  }
  async function feedback(rid, claimId, label, btn) {
    btn.disabled = true;
    try { await api("POST", "/api/v1/feedback", { request_id: rid, claim_id: claimId, label }); btn.textContent = "thanks"; } catch (e) { btn.textContent = "failed"; }
  }
  return { $, esc, api, toast, state, headers, register, route, chip, renderBanner, renderAnswer, renderClaims, renderSources, renderAnalysis, feedback, loadRecent, fmtPct, tierCls, labelCls };
})();
