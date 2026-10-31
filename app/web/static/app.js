/* Renders the API response. Nothing here decides anything: every label, action and number
   comes from POST /api/v1/query. */
(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const ACTION_TEXT = {
    ANSWER: "Answer supported by the cited evidence",
    ANSWER_WITH_CAUTION: "Answer delivered with caution: some claims were removed or sources disagree",
    CLARIFY: "The question needs clarification before it can be answered from the evidence",
    ABSTAIN: "No answer: the evidence does not support a reliable answer",
    ESCALATE: "Out of scope for this tool",
  };
  try { const t = sessionStorage.getItem("litbiorag.token"); if (t) $("token").value = t; } catch (e) {}

  function chip(text, cls) { return `<span class="chip ${cls || ""}">${esc(text)}</span>`; }
  function labelCls(l) { return l === "supported" ? "ok" : l === "contradicted" ? "bad" : "warn"; }

  function headers() {
    const h = { "Content-Type": "application/json" };
    const t = $("token").value.trim();
    if (t) { h["Authorization"] = "Bearer " + t; try { sessionStorage.setItem("litbiorag.token", t); } catch (e) {} }
    return h;
  }

  async function ask(ev) {
    ev.preventDefault();
    const q = $("q").value.trim();
    const courses = $("courses").value.split(",").map((s) => s.trim()).filter(Boolean);
    $("go").disabled = true; $("status").textContent = "retrieving, generating, verifying…";
    try {
      const r = await fetch("/api/v1/query", { method: "POST", headers: headers(),
        body: JSON.stringify({ question: q, course_ids: courses.length ? courses : null }) });
      const body = await r.json();
      if (!r.ok) { $("status").innerHTML = `<span class="err">${esc(body.error || r.status)}</span>`; return; }
      render(body); $("status").textContent = `request ${body.request_id} · ${body.timings_ms.total} ms`;
    } catch (e) { $("status").innerHTML = `<span class="err">request failed</span>`; }
    finally { $("go").disabled = false; }
  }

  function render(b) {
    $("out").hidden = false;
    const conf = b.confidence == null ? "n/a" : (b.confidence_calibrated ? b.confidence.toFixed(2) : `${b.confidence.toFixed(2)} (uncalibrated)`);
    $("banner").className = "banner " + b.action;
    $("banner").innerHTML = `<div><b>${esc(b.action.replace(/_/g, " "))}</b> — ${esc(ACTION_TEXT[b.action] || "")}
      <small>Deciding rule ${esc(b.rule)}: ${esc(b.reason)} · confidence ${esc(conf)} · evidence sufficiency ${b.sufficiency == null ? "n/a" : b.sufficiency.toFixed(2)}</small></div>`;
    $("answer").innerHTML = `<p>${esc(b.answer)}</p>`;
    $("removed").innerHTML = b.removed_claims.length
      ? `<details><summary>${b.removed_claims.length} claim(s) removed from this answer</summary><ul>${b.removed_claims.map((c) => `<li class="muted">${esc(c)}</li>`).join("")}</ul></details>` : "";
    const e = b.entities;
    const subs = b.claims.filter((c) => c.substitution);
    $("entities").innerHTML = e.asked.length
      ? `Asked about: ${e.asked.map((x) => chip(x, e.missing_in_evidence.includes(x) ? "bad" : "ok")).join("")}`
        + (e.missing_in_evidence.length ? ` · not found in the evidence: ${e.missing_in_evidence.map(esc).join(", ")}` : "")
        + (subs.length ? `<div class="err">Entity substitution detected: the answer named <b>${esc(subs[0].substitution.claim_entity)}</b> where the question asked about <b>${esc(subs[0].substitution.asked_entity)}</b> (${esc(subs[0].substitution.reason || "near neighbour")}).</div>` : "")
      : "";
    $("conflicts-card").hidden = !b.conflicts.length;
    $("claims-card").hidden = !b.claims.length;
    $("sources-card").hidden = !b.sources.length;
    $("conflicts").innerHTML = b.conflicts.map((c) => `<div><b>Supports</b><br>${c.supporting_ids.map((i) => chip(i)).join("")}</div><div><b>Contradicts</b><br>${c.contradicting_ids.map((i) => chip(i)).join("")}<p class="muted">${esc(c.description)}</p></div>`).join("");
    const srcByPid = Object.fromEntries(b.sources.map((s) => [s.passage_id, s]));
    $("claims").innerHTML = b.claims.map((c) => {
      const span = c.evidence_span && srcByPid[c.evidence_span.passage_id];
      const cspan = c.contradicting_span && srcByPid[c.contradicting_span.passage_id];
      return `<div class="claim ${c.removed ? "removed" : ""}">
        ${chip(c.label, labelCls(c.label))}${c.critical ? chip("critical") : ""}${c.removed ? chip("removed", "warn") : ""}${c.off_target ? chip("off-target", "warn") : ""}
        ${c.fabricated_ids.length ? chip("fabricated citation", "bad") : ""}${c.numeric_mismatches.length ? chip("number mismatch", "bad") : ""}${c.entity_missing.length ? chip("entity not in evidence", "bad") : ""}
        <div class="txt">${esc(c.text)}</div>
        <div class="muted">cites ${c.citations.map((i) => chip(i, c.supporting_ids.includes(i) ? "ok" : (c.fabricated_ids.includes(i) ? "bad" : ""))).join("") || "nothing"} · support ${c.support.toFixed(2)} · contradiction ${c.contradiction.toFixed(2)}</div>
        ${span ? `<blockquote>${esc(span.text)}<br><span class="mono">${esc(span.citation_id)} chars ${span.char_start}–${span.char_end}</span></blockquote>` : ""}
        ${cspan ? `<blockquote class="err">contradicting: ${esc(cspan.text)}<br><span class="mono">${esc(cspan.citation_id)}</span></blockquote>` : ""}
        <div class="fb">${chip("Is this claim right?")}<button class="ghost" data-claim="${esc(c.id)}" data-label="correct">yes</button> <button class="ghost" data-claim="${esc(c.id)}" data-label="incorrect">no</button></div>
      </div>`; }).join("") || `<p class="muted">No claims were generated.</p>`;
    $("sources").innerHTML = b.sources.map((s) => `<div class="src"><div class="id">${esc(s.citation_id)}<br>#${s.rank}</div>
      <div>${chip(s.tier.replace("_", " "), s.tier === "peer_reviewed" ? "ok" : "warn")}${s.year ? chip(s.year) : ""}<b>${esc(s.display_source)}</b>${s.journal ? ` <span class="muted">· ${esc(s.journal)}</span>` : ""}
      ${s.url ? ` <a href="${esc(s.url)}" target="_blank" rel="noopener">open</a>` : ""}<div class="muted">relevance ${s.retrieval_score.toFixed(3)} · ${esc(s.text.slice(0, 220))}${s.text.length > 220 ? "…" : ""}</div></div></div>`).join("");
    const v = b.verification;
    $("report").innerHTML = Object.entries(v).filter(([k, x]) => x !== null && typeof x !== "object").map(([k, x]) => `<dt>${esc(k)}</dt><dd>${esc(typeof x === "number" ? x.toFixed(3) : x)}</dd>`).join("")
      + (b.usage ? `<dt>tokens</dt><dd>${b.usage.prompt_tokens} in / ${b.usage.completion_tokens} out${b.usage.exact ? "" : " (estimated)"}</dd>` : "");
    $("trace").textContent = b.trace.join("\n");
    const m = b.versions;
    const models = Object.entries(m.models || {}).map(([r, s]) => `${r}: ${s.name}`).join(" · ");
    $("meta").innerHTML = `${esc(b.disclosure)}<br>profile <b>${esc(m.profile)}</b> · ${esc(models)}<br>corpus bundle ${esc(m.bundle_id)} (${esc(m.bundle_created_at || "n/a")}) · prompt ${esc(m.prompt_id)} · gate calibration ${esc(m.gate_calibrated_on)} · API ${esc(m.api_version)} · request ${esc(b.request_id)}`;
    document.querySelectorAll(".fb button").forEach((btn) => btn.addEventListener("click", () => feedback(b.request_id, btn.dataset.claim, btn.dataset.label, btn)));
  }

  async function feedback(rid, claimId, label, btn) {
    btn.disabled = true;
    try {
      const r = await fetch("/api/v1/feedback", { method: "POST", headers: headers(),
        body: JSON.stringify({ request_id: rid, claim_id: claimId, label }) });
      btn.textContent = r.ok ? "thanks" : "failed";
    } catch (e) { btn.textContent = "failed"; }
  }

  $("ask").addEventListener("submit", ask);
})();
