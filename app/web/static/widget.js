/* AEOS Medical embeddable chat widget.
   <script src="https://HOST/static/widget.js" data-api="https://HOST" data-key="lbr_…"
           data-scope="uploaded" data-kb="kb_…" data-title="Ask the course"></script>
   Renders a compact chat box into the element with id "aeos-widget" (created if absent). It
   renders only what the API returns: action banner, answer, citations, review notice. */
(function () {
  const s = document.currentScript; const api = (s.dataset.api || "").replace(/\/$/, "");
  const key = s.dataset.key || ""; const token = s.dataset.token || "";
  const scope = s.dataset.scope || "uploaded"; const kb = s.dataset.kb || ""; const courses = s.dataset.courses || "";
  const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let host = document.getElementById("aeos-widget");
  if (!host) { host = document.createElement("div"); host.id = "aeos-widget"; document.body.appendChild(host); }
  host.innerHTML = `<style>
    #aeos-widget{font:14px/1.5 system-ui,sans-serif;max-width:560px;border:1px solid #d9e0ea;border-radius:12px;padding:12px;background:#fff;color:#101828}
    #aeos-widget .t{display:flex;gap:8px;margin-bottom:8px}#aeos-widget input{flex:1;padding:8px;border:1px solid #d9e0ea;border-radius:8px}
    #aeos-widget button{padding:8px 12px;border:0;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer}
    #aeos-widget .b{border-left:5px solid #60a5fa;background:#f1f4f9;padding:8px 10px;border-radius:8px;margin:8px 0}
    #aeos-widget .ANSWER{border-color:#15803d}#aeos-widget .ANSWER_WITH_CAUTION{border-color:#b45309}#aeos-widget .ABSTAIN,#aeos-widget .ESCALATE{border-color:#b91c1c}
    #aeos-widget .c{display:inline-block;background:#e9eef6;border-radius:5px;padding:0 5px;font-size:12px;margin:0 2px}
    #aeos-widget .m{color:#5b6b82;font-size:12px}</style>
    <div class="t"><input id="aeos-q" placeholder="${esc(s.dataset.title || "Ask a question about the course")}"><button id="aeos-go">Ask</button></div>
    <div id="aeos-out"></div><div class="m">AI-generated summary of the course material; verified claim by claim; not medical advice.</div>`;
  const out = host.querySelector("#aeos-out"); const q = host.querySelector("#aeos-q"); const go = host.querySelector("#aeos-go");
  let chatId = null;
  async function ask() {
    const question = q.value.trim(); if (!question) return;
    go.disabled = true; out.insertAdjacentHTML("afterbegin", `<div class="b"><b>You:</b> ${esc(question)}</div>`);
    const h = { "Content-Type": "application/json" }; if (token) h["Authorization"] = "Bearer " + token; else if (key) h["X-API-Key"] = key;
    try {
      if (!chatId && (token || key)) { try { const c = await (await fetch(api + "/api/v1/chats", { method: "POST", headers: h, body: "{}" })).json(); chatId = c.chat_id || null; } catch (e) {} }
      const body = { question, scope }; if (kb) body.kb_ids = kb.split(","); if (courses) body.course_ids = courses.split(","); if (chatId) body.chat_id = chatId;
      const r = await fetch(api + "/api/v1/query", { method: "POST", headers: h, body: JSON.stringify(body) }); const d = await r.json();
      if (!r.ok) { out.insertAdjacentHTML("afterbegin", `<div class="b ABSTAIN">${esc(d.error || r.status)}</div>`); return; }
      const idx = {}; d.sources.forEach((src, i) => { idx[src.citation_id] = i + 1; });
      const cites = d.sources.map((src, i) => `<div class="m">[${i + 1}] ${esc(src.display_source)} <span class="c">${esc(src.tier.replace("_", " "))}</span></div>`).join("");
      const review = d.review ? `<div class="m">${esc(d.review.message)}</div>` : "";
      out.insertAdjacentHTML("afterbegin", `<div class="b ${esc(d.action)}"><b>${esc(d.action.replace(/_/g, " "))}</b> <span class="m">${esc(d.reason)}</span><p>${esc(d.answer).replace(/\[((?:SLIDE|FILE|PMID|DOI):[^\]]+)\]/g, (m, id) => idx[id] ? `<span class="c">[${idx[id]}]</span>` : m)}</p>${cites}${review}</div>`);
      q.value = "";
    } catch (e) { out.insertAdjacentHTML("afterbegin", `<div class="b ABSTAIN">request failed</div>`); }
    finally { go.disabled = false; }
  }
  go.addEventListener("click", ask); q.addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
})();
