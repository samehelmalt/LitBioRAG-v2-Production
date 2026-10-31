/* Page modules. Each renders from API responses only. */
(function (A) {
  const { $, esc, api, toast, state, chip } = A;
  const kv = (o) => `<dl class="kv">${Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === "object" && v !== null ? JSON.stringify(v) : v)}</dd>`).join("")}</dl>`;
  const when = (t) => (t ? new Date(t * 1000).toLocaleString() : "");
  const needSignin = (main, what) => { main.innerHTML = `<div class="card">Sign in to ${esc(what)}. <button class="btn" onclick="document.getElementById('signin').click()">Sign in</button></div>`; };

  // ================================================================= chat ==================
  A.register("chat", async (main, rest) => {
    const chatId = rest[0] || null;
    state.chatId = chatId;
    main.innerHTML = `
      <section>
        <div class="card">
          <div class="tabs"><button class="active" data-tab="chat">Chat</button><button data-tab="documents">Documents</button><button data-tab="citations">Citations</button><button data-tab="analysis">Analysis</button></div>
          <div id="tab-chat">
            <div id="thread"></div>
            <form class="qbox" id="ask">
              <textarea id="q" placeholder="Ask a biomedical question, e.g. What are the latest treatment approaches for non-small cell lung cancer?" required minlength="3"></textarea>
              <div class="row">
                <label class="check"><input type="checkbox" id="sc-uploaded" ${state.scope.uploaded ? "checked" : ""}> my uploaded files</label>
                <label class="check"><input type="checkbox" id="sc-indexed" ${state.scope.indexed ? "checked" : ""}> indexed corpus (PubMed, bioRxiv/medRxiv)</label>
                <input type="text" id="courses" placeholder="course ids (optional)" style="min-width:160px">
                <button class="btn" id="go" type="submit">Ask ➤</button><span id="status" class="muted small"></span>
              </div>
            </form>
            <div class="steps" id="steps"></div>
          </div>
          <div id="tab-documents" hidden></div><div id="tab-citations" hidden></div><div id="tab-analysis" hidden></div>
        </div>
      </section>
      <aside>
        <div class="card"><h3>Source Documents <span id="srccount" class="chip"></span></h3><div id="sources" class="muted small">Sources of the latest answer appear here.</div></div>
        <div class="card"><h3>Analysis</h3><div id="analysis" class="muted small">Verification of the latest answer appears here.</div></div>
      </aside>`;
    let last = null;
    const tabs = main.querySelectorAll(".tabs button");
    tabs.forEach((b) => b.addEventListener("click", () => { tabs.forEach((x) => x.classList.toggle("active", x === b)); ["chat", "documents", "citations", "analysis"].forEach((t) => { $("tab-" + t).hidden = t !== b.dataset.tab; }); }));
    main.addEventListener("click", (e) => { const c = e.target.closest("[data-cite]"); if (c) { const el = document.getElementById("src-" + c.dataset.cite); if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.style.outline = "2px solid var(--accent2)"; setTimeout(() => { el.style.outline = ""; }, 1500); } } const fb = e.target.closest("button.fb"); if (fb && last) A.feedback(last.request_id, fb.dataset.claim, fb.dataset.label, fb); });

    if (chatId) {
      try {
        const chat = await api("GET", `/api/v1/chats/${chatId}`);
        chat.messages.forEach((m) => { appendQ(m.question); if (m.response) { appendA(m.response); last = m.response; } });
        if (last) showPanels(last);
      } catch (e) { toast(e.message); }
    }
    function appendQ(q) { $("thread").insertAdjacentHTML("beforeend", `<div class="bubble q">${esc(q)}</div>`); }
    function appendA(b) { $("thread").insertAdjacentHTML("beforeend", A.renderAnswer(b)); }
    function showPanels(b) {
      $("sources").innerHTML = A.renderSources(b, true); $("srccount").textContent = b.sources.length;
      $("analysis").innerHTML = A.renderAnalysis(b);
      $("tab-documents").innerHTML = A.renderSources(b, false);
      $("tab-citations").innerHTML = A.renderClaims(b);
      $("tab-analysis").innerHTML = A.renderAnalysis(b) + `<details><summary class="muted small">full verification report</summary>${kv(b.verification)}</details>`;
    }
    const STAGES = { scope_check: "Checking scope", retrieving: "Searching knowledge base", retrieved: "Retrieving relevant documents", generating: "Generating grounded answer", verifying: "Verifying claims against evidence", deciding: "Deciding delivery" };
    $("ask").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const q = $("q").value.trim(); if (!q) return;
      const up = $("sc-uploaded").checked, ix = $("sc-indexed").checked;
      state.scope = { uploaded: up, indexed: ix };
      if (!up && !ix) { toast("choose at least one retrieval scope"); return; }
      const scope = up && ix ? "both" : up ? "uploaded" : "indexed";
      const courses = $("courses").value.split(",").map((s) => s.trim()).filter(Boolean);
      if (state.principal && !state.chatId) {
        try { const c = await api("POST", "/api/v1/chats", {}); state.chatId = c.chat_id; history.replaceState(null, "", `#/chat/${c.chat_id}`); } catch (e) {}
      }
      appendQ(q); $("q").value = ""; $("go").disabled = true; $("steps").innerHTML = "";
      const seen = [];
      const params = new URLSearchParams({ question: q, scope });
      if (state.chatId) params.set("chat_id", state.chatId);
      if (courses.length) params.set("course_ids", courses.join(","));
      try {
        const r = await fetch("/api/v1/query/stream?" + params.toString(), { headers: A.headers(false) });
        if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || `HTTP ${r.status}`); }
        const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
        while (true) {
          const { value, done } = await reader.read(); if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n\n")) >= 0) {
            const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
            const kind = (chunk.match(/^event: (\w+)/m) || [])[1]; const data = JSON.parse(chunk.split("\ndata: ")[1] || "{}");
            if (kind === "stage") { seen.push(data.stage); $("steps").innerHTML = seen.map((s, i) => `<div class="${i === seen.length - 1 ? "run" : "done"}">${esc(STAGES[s] || s)}${s === "retrieved" ? ` (${data.documents} documents)` : ""}</div>`).join(""); }
            else if (kind === "result") { last = data; appendA(data); showPanels(data); $("steps").innerHTML = ""; $("status").textContent = `request ${data.request_id} · ${data.timings_ms.total} ms`; A.loadRecent(); }
            else if (kind === "error") { $("thread").insertAdjacentHTML("beforeend", `<div class="bubble err">${esc(data.error)}</div>`); $("steps").innerHTML = ""; }
          }
        }
      } catch (e) { $("thread").insertAdjacentHTML("beforeend", `<div class="bubble err">${esc(e.message)}</div>`); }
      finally { $("go").disabled = false; }
    });
  });

  // ================================================================= knowledge base ========
  A.register("kb", async (main) => {
    if (!state.principal) return needSignin(main, "manage knowledge bases");
    const kbs = await api("GET", "/api/v1/kb");
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Knowledge Base</h2><p>Your uploaded material, indexed with page / slide / sheet-level citations. Search it alone or together with the indexed corpus.</p></div></div>
      <div class="card"><form id="newkb" class="row"><input type="text" id="kbname" placeholder="new knowledge base name" required><input type="text" id="kbdesc" placeholder="description (optional)"><button class="btn">Create</button></form></div>
      ${kbs.length ? kbs.map((k) => `<div class="card"><h3>${esc(k.name)} <span class="chip">${esc(k.kb_id)}</span> <span class="chip">v${k.version}</span></h3>
        <div class="small muted">${esc(k.description || "")} · owner ${esc(k.owner)} · shared with ${k.shared_with.length ? k.shared_with.map(esc).join(", ") : "nobody"} · created ${when(k.created_at)}</div>
        <table class="table"><tr><th>file</th><th>kind</th><th>units</th><th>status</th><th>uploaded</th><th></th></tr>
        ${k.files.map((f) => `<tr><td>${esc(f.file_name)}<br><span class="mono muted">${esc(f.file_id)}</span></td><td>${esc(f.kind)}</td><td>${f.n_units}</td><td>${chip(f.status, f.status === "indexed" ? "ok" : f.status === "failed" ? "bad" : "warn")}${f.error ? `<div class="small err">${esc(f.error)}</div>` : ""}</td><td class="small">${when(f.uploaded_at)} by ${esc(f.uploaded_by)}</td><td><button class="btn ghost small" data-del="${esc(k.kb_id)}/${esc(f.file_id)}">remove</button></td></tr>`).join("") || '<tr><td colspan="6" class="muted">no files yet — use Upload Documents</td></tr>'}</table>
        <div class="row" style="margin-top:8px"><a class="btn" href="#/upload/${esc(k.kb_id)}">Upload files</a><button class="btn ghost" data-share="${esc(k.kb_id)}">Share…</button><button class="btn ghost danger" data-delkb="${esc(k.kb_id)}">Delete knowledge base</button></div></div>`).join("") : '<div class="card muted">No knowledge bases yet.</div>'}`;
    $("newkb").addEventListener("submit", async (e) => { e.preventDefault(); try { await api("POST", "/api/v1/kb", { name: $("kbname").value, description: $("kbdesc").value }); A.route(); } catch (err) { toast(err.message); } });
    main.addEventListener("click", async (e) => {
      const d = e.target.closest("[data-del]"); const s = e.target.closest("[data-share]"); const k = e.target.closest("[data-delkb]");
      try {
        if (d && confirm("Remove this file from the index?")) { await api("DELETE", `/api/v1/kb/${d.dataset.del.replace("/", "/files/")}`); A.route(); }
        if (s) { const subs = prompt("Share with user ids (comma-separated)"); if (subs !== null) { await api("POST", `/api/v1/kb/${s.dataset.share}/share`, { subs: subs.split(",").map((x) => x.trim()).filter(Boolean) }); A.route(); } }
        if (k && confirm("Delete the whole knowledge base and its index?")) { await api("DELETE", `/api/v1/kb/${k.dataset.delkb}`); A.route(); }
      } catch (err) { toast(err.message); }
    });
  });

  // ================================================================= upload ================
  A.register("upload", async (main, rest) => {
    if (!state.principal) return needSignin(main, "upload documents");
    const kbs = await api("GET", "/api/v1/kb");
    const mine = kbs.filter((k) => k.owner === state.principal.sub || state.principal.roles.includes("admin"));
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Upload Documents</h2><p>PDF, Word, Excel, PowerPoint, text and Markdown. Each page, section, sheet block or slide becomes a citable unit.</p></div></div>
      <div class="card"><div class="row"><label>Knowledge base <select id="kbsel" class="inp">${mine.map((k) => `<option value="${esc(k.kb_id)}" ${rest[0] === k.kb_id ? "selected" : ""}>${esc(k.name)}</option>`).join("")}</select></label>${mine.length ? "" : '<span class="muted">create one on the Knowledge Base page first</span>'}</div>
      <div class="drop" id="drop" style="margin-top:12px">Drop files here or <label style="color:var(--accent2);cursor:pointer">browse<input type="file" id="files" multiple hidden accept=".pdf,.docx,.xlsx,.pptx,.txt,.md"></label></div>
      <table class="table" id="jobs" style="margin-top:12px"></table></div>`;
    const drop = $("drop");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));
    $("files").addEventListener("change", (e) => upload(e.target.files));
    async function upload(files) {
      const kb = $("kbsel").value; if (!kb) { toast("no knowledge base selected"); return; }
      for (const f of files) {
        const row = document.createElement("tr"); row.innerHTML = `<td>${esc(f.name)}</td><td>${(f.size / 1024).toFixed(0)} KB</td><td class="muted">uploading…</td>`; $("jobs").appendChild(row);
        const fd = new FormData(); fd.append("file", f); fd.append("background", "false");
        try {
          const r = await fetch(`/api/v1/kb/${kb}/upload`, { method: "POST", headers: A.headers(false), body: fd });
          const d = await r.json();
          row.lastElementChild.innerHTML = r.ok ? `${chip(d.status, d.status === "indexed" ? "ok" : d.status === "failed" ? "bad" : "warn")} ${d.n_units || 0} units${d.error ? ` <span class="err">${esc(d.error)}</span>` : ""}` : `<span class="err">${esc(d.error)}</span>`;
        } catch (err) { row.lastElementChild.innerHTML = `<span class="err">${esc(err.message)}</span>`; }
      }
    }
  });

  // ================================================================= tools =================
  A.register("tools", async (main) => {
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Research Tools</h2><p>The same checks the pipeline runs, on your own text.</p></div></div>
      <div class="two"><div class="card"><h3>Confusable-entity check</h3><p class="small muted">Finds genes, drugs and conditions in a text and flags near neighbours of the entities in your question (MAP3K3 vs MAP3K7, atorvastatin vs simvastatin, hypo- vs hyperkalemia).</p>
        <input class="inp" id="t-q" type="text" placeholder="your question" style="width:100%;margin-bottom:6px"><textarea class="inp" id="t-text" style="width:100%;min-height:90px" placeholder="text to check (an answer, an abstract…)"></textarea><div class="row" style="margin-top:6px"><button class="btn" id="t-run">Check</button></div><div id="t-out"></div></div>
      <div class="card"><h3>Claim vs passage</h3><p class="small muted">Scores one claim against one passage with the served verifier and the entity check.</p>
        <textarea class="inp" id="v-claim" style="width:100%;min-height:50px" placeholder="claim"></textarea><textarea class="inp" id="v-pass" style="width:100%;min-height:90px;margin-top:6px" placeholder="passage"></textarea><div class="row" style="margin-top:6px"><button class="btn" id="v-run">Verify</button></div><div id="v-out"></div></div></div>`;
    $("t-run").addEventListener("click", async () => { try { const r = await api("POST", "/api/v1/tools/entities", { text: $("t-text").value, question: $("t-q").value }); $("t-out").innerHTML = `<p>entities: ${r.mentions.map((m) => chip(`${m.surface} · ${m.kind}`)).join("") || "none"}</p>${r.confusable.length ? `<div class="err">Confusable: ${r.confusable.map((c) => `<b>${esc(c.text_entity)}</b> vs asked <b>${esc(c.asked_entity)}</b> (${esc(c.reason)})`).join("; ")}</div>` : '<p class="muted">no confusable entities</p>'}`; } catch (e) { toast(e.message); } });
    $("v-run").addEventListener("click", async () => { try { const r = await api("POST", "/api/v1/tools/verify", { claim: $("v-claim").value, passage: $("v-pass").value }); $("v-out").innerHTML = kv({ entail: r.nli.entail.toFixed(3), contradict: r.nli.contradict.toFixed(3), neutral: r.nli.neutral.toFixed(3), entities_missing: r.entities_missing.join(", ") || "none", verifier: r.verifier.name }); } catch (e) { toast(e.message); } });
  });

  // ================================================================= evaluations ===========
  A.register("evaluations", async (main) => {
    const m = await api("GET", "/api/v1/metrics");
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Evaluations</h2><p>Benchmark runs on held-out splits and adversarial suites. Numbers here are measured; nothing is computed per answer.</p></div></div>
      ${m.status === "NOT YET MEASURED" ? '<div class="card"><b>NOT YET MEASURED.</b> No benchmark run has been recorded for this bundle and profile. Runs appear here once <span class="mono">scripts/run_retrieval_baseline.py</span>, <span class="mono">run_answer_baseline.py</span> and <span class="mono">run_adversarial.py</span> have been executed with real models.</div>'
        : Object.entries(m).map(([k, v]) => `<div class="card"><h3>${esc(k)}</h3>${kv(v)}</div>`).join("")}`;
  });

  // ================================================================= datasets ==============
  A.register("datasets", async (main) => {
    const d = await api("GET", "/api/v1/datasets");
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Datasets</h2><p>Evaluation splits and the human labels collected so far.</p></div></div>
      <div class="two"><div class="card"><h3>Split manifest ${esc(d.splits.version || "")}</h3>${d.splits.counts ? kv({ ...d.splits.counts, pinned: d.splits.pinned, seed: d.splits.seed }) : '<span class="muted">no manifest</span>'}<p class="small muted">${esc(d.splits.notes || "")}</p></div>
      <div class="card"><h3>Human labels</h3>${kv({ "in-product feedback": d.labels.feedback, "reviewer decisions": d.labels.review_labels })}<p class="small muted">Labels feed gate calibration and the false-PASS measurement once enough exist.</p></div></div>`;
  });

  // ================================================================= models ================
  A.register("models", async (main) => {
    const m = await api("GET", "/api/v1/models");
    let admin = null; if (state.principal && state.principal.roles.includes("admin")) { try { admin = await api("GET", "/api/v1/admin/profiles"); } catch (e) {} }
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Model Settings</h2><p>Models are configuration (configs/models.yaml). Swapping a model never changes code; calibrated cut-offs belong to the model they were fitted on.</p></div></div>
      <div class="card"><h3>Active profile: ${esc(m.profile)}</h3><table class="table"><tr><th>role</th><th>model</th><th>family</th><th>quantization</th><th>endpoint</th><th>calibrated</th></tr>
      ${Object.entries(m.models).map(([r, s]) => `<tr><td>${esc(r)}</td><td class="mono">${esc(s.name)}</td><td>${esc(s.family)}</td><td>${esc(s.quantization || "—")}</td><td class="mono">${esc(s.endpoint || "—")}</td><td>${s.calibrated ? chip("yes", "ok") : chip("no", "warn")}</td></tr>`).join("")}</table>
      ${kv({ prompt: m.prompt_id, bundle: m.bundle_id, gate_calibrated_on: m.gate_calibrated_on, "gate costs": m.gate.costs })}</div>
      ${admin ? `<div class="card"><h3>Switch profile (admin)</h3><div class="row"><select id="prof" class="inp">${Object.keys(admin.profiles).map((p) => `<option ${p === admin.active ? "selected" : ""}>${esc(p)}</option>`).join("")}</select><button class="btn" id="switch">Apply</button><span class="muted small">pre-flight runs first; the service keeps the current profile on failure</span></div>
        ${Object.entries(admin.profiles).map(([p, v]) => `<details><summary>${esc(p)} — ${esc(v.description || "").slice(0, 120)}</summary>${kv(v.models)}</details>`).join("")}</div>` : ""}`;
    if (admin) $("switch").addEventListener("click", async () => { try { const r = await api("POST", "/api/v1/admin/profile", { profile: $("prof").value }); toast(`active profile: ${r.active}`); A.route(); } catch (e) { toast(e.message, 8000); } });
  });

  // ================================================================= api access ============
  A.register("api-access", async (main) => {
    if (!state.principal) return needSignin(main, "manage API keys");
    let keys = []; try { keys = await api("GET", "/api/v1/keys"); } catch (e) { keys = []; }
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>API Access</h2><p>Keys act as you (same knowledge bases, no admin roles). The full key is shown once.</p></div></div>
      <div class="card"><form id="newkey" class="row"><input type="text" id="keyname" placeholder="key name (e.g. notebook)" required><button class="btn">Issue key</button></form><div id="shown"></div></div>
      <div class="card"><table class="table"><tr><th>prefix</th><th>name</th><th>created</th><th>last used</th><th>status</th><th></th></tr>${keys.map((k) => `<tr><td class="mono">${esc(k.prefix)}…</td><td>${esc(k.name)}</td><td>${when(k.created_at)}</td><td>${when(k.last_used_at) || "—"}</td><td>${k.revoked_at ? chip("revoked", "bad") : chip("active", "ok")}</td><td>${k.revoked_at ? "" : `<button class="btn ghost small" data-rev="${esc(k.prefix)}">revoke</button>`}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">no keys</td></tr>'}</table></div>
      <div class="card"><h3>Use it</h3><pre class="log">curl -X POST ${esc(location.origin)}/api/v1/query -H "X-API-Key: lbr_…" -H "Content-Type: application/json" \\
  -d '{"question": "What does MAP3K3 do in NF-kappaB signalling?", "scope": "both"}'</pre><a href="/api/docs" target="_blank">OpenAPI docs</a></div>`;
    $("newkey").addEventListener("submit", async (e) => { e.preventDefault(); try { const k = await api("POST", "/api/v1/keys", { name: $("keyname").value }); $("shown").innerHTML = `<div class="banner ANSWER">Your new key (copy it now, it is not shown again):<br><span class="mono">${esc(k.key)}</span></div>`; } catch (err) { toast(err.message); } });
    main.addEventListener("click", async (e) => { const b = e.target.closest("[data-rev]"); if (b) { try { await api("DELETE", `/api/v1/keys/${b.dataset.rev}`); A.route(); } catch (err) { toast(err.message); } } });
  });

  // ================================================================= review ================
  A.register("review", async (main, rest) => {
    if (!state.principal) return needSignin(main, "see review cases");
    const isReviewer = state.principal.roles.some((r) => r === "reviewer" || r === "admin");
    const view = rest[0] || (isReviewer ? "queue" : "mine");
    if (view === "mine") {
      const cases = await api("GET", "/api/v1/my/cases");
      main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>My cases</h2><p>Answers the safety gate withheld and sent to a human reviewer.</p></div></div>${isReviewer ? '<p><a href="#/review/queue">Reviewer queue</a></p>' : ""}
        ${cases.map((c) => `<div class="card"><div class="small muted">${when(c.created_at)} · ${chip(c.status, c.status === "released" ? "ok" : c.status === "pending" ? "warn" : "bad")} machine action ${esc(c.action)} (${esc(c.rule)})</div><p><b>${esc(c.question)}</b></p>
          ${c.status === "released" ? `<div class="banner ANSWER">${chip("human-reviewed", "ok")} reviewed by ${esc(c.reviewer)} on ${when(c.decided_at)}<p>${esc(c.released_answer)}</p></div>` : c.status === "blocked" ? `<div class="banner ABSTAIN">A reviewer confirmed that no answer can be released. ${esc(c.reviewer_notes || "")}</div>` : '<div class="muted">pending review</div>'}</div>`).join("") || '<div class="card muted">No cases.</div>'}`;
      return;
    }
    if (!isReviewer) { main.innerHTML = '<div class="card err">Reviewer role required.</div>'; return; }
    if (view === "case") {
      const c = await api("GET", `/api/v1/review/${rest[1]}`);
      const r = c.response;
      main.innerHTML = `<p><a href="#/review/queue">← queue</a></p><div class="card"><h3>Case ${esc(c.case_id)} ${chip(c.status, c.status === "pending" ? "warn" : c.status === "released" ? "ok" : "bad")}</h3>
        <div class="small muted">asked by ${esc(c.sub)} · ${when(c.created_at)} · request ${esc(c.request_id)}</div><p><b>${esc(c.question)}</b></p>${A.renderBanner(r)}
        <h3>Machine answer (withheld)</h3><p>${esc(r.answer)}</p><h3>Claims</h3>${A.renderClaims(r)}<h3>Sources</h3>${A.renderSources(r, false)}</div>
        ${c.status === "pending" ? `<div class="card"><h3>Decision</h3><p class="small muted">Release the machine's supported claims, a subset, or your own edited text. Edited text is re-verified against the same evidence; flagged claims need explicit confirmation.</p>
          <div>${r.claims.map((cl) => `<label class="check"><input type="checkbox" class="keep" value="${esc(cl.id)}" ${cl.label === "supported" && !cl.removed ? "checked" : ""}> ${chip(cl.label, A.labelCls(cl.label))} ${esc(cl.text)}</label><br>`).join("")}</div>
          <textarea class="inp" id="edited" style="width:100%;min-height:80px;margin-top:8px" placeholder="edited answer (optional; replaces the kept claims)"></textarea>
          <input class="inp" id="notes" type="text" placeholder="notes (required for release)" style="width:100%;margin-top:6px">
          <div class="row" style="margin-top:8px"><button class="btn" id="release">Release</button><button class="btn danger" id="block">Keep blocked</button></div><div id="reverify"></div></div>` : `<div class="card">${kv({ decision: c.status, reviewer: c.reviewer, decided: when(c.decided_at), notes: c.reviewer_notes, released_answer: c.released_answer })}</div>`}`;
      if (c.status !== "pending") return;
      const decide = async (decision, confirm) => {
        const kept = [...main.querySelectorAll(".keep:checked")].map((x) => x.value);
        try {
          const out = await api("POST", `/api/v1/review/${c.case_id}/decision`, { decision, edited_answer: $("edited").value || null, kept_claim_ids: kept, notes: $("notes").value, confirm: !!confirm });
          toast(`case ${out.status}`); location.hash = "#/review/queue";
        } catch (e) {
          if (e.status === 409 && e.data && e.data.reverification) {
            const rv = e.data.reverification;
            $("reverify").innerHTML = `<div class="banner ABSTAIN"><b>Re-verification flagged ${rv.flagged_claims.length} claim(s)</b> (support ${A.fmtPct(rv.entailment_rate)}, contradiction ${A.fmtPct(rv.contradiction_rate)}${rv.substitution_detected ? ", entity substitution" : ""}).<br>${rv.verdicts.filter((v) => v.label !== "supported").map((v) => `${chip(v.label, A.labelCls(v.label))} ${esc(v.text)}`).join("<br>")}<div class="row" style="margin-top:8px"><button class="btn danger" id="confirm">Release anyway (recorded)</button></div></div>`;
            $("confirm").addEventListener("click", () => decide("release", true));
          } else toast(e.message, 6000);
        }
      };
      $("release").addEventListener("click", () => decide("release", false));
      $("block").addEventListener("click", () => decide("block", false));
      return;
    }
    const status = rest[1] || "pending";
    const [cases, stats] = await Promise.all([api("GET", `/api/v1/review/queue?status=${status}`), api("GET", "/api/v1/review/stats")]);
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Review Queue</h2><p>Blocked and escalated answers awaiting a human decision. The gate decided first; you can release or keep blocked, never bypass verification silently.</p></div></div>
      <div class="row" style="margin-bottom:10px">${["pending", "released", "blocked", "expired"].map((s) => `<a class="chip ${s === status ? "acc" : ""}" href="#/review/queue/${s}">${s} ${stats[s] || 0}</a>`).join("")}<a class="chip" href="#/review/mine">my cases</a></div>
      <div class="card"><table class="table"><tr><th>when</th><th>question</th><th>action · rule</th><th>asked by</th><th></th></tr>${cases.map((c) => `<tr><td class="small">${when(c.created_at)}</td><td>${esc(c.question)}</td><td>${chip(c.action, "bad")}<span class="mono">${esc(c.rule)}</span><div class="small muted">${esc(c.reason)}</div></td><td>${esc(c.sub)}</td><td><a class="btn ghost small" href="#/review/case/${esc(c.case_id)}">open</a></td></tr>`).join("") || `<tr><td colspan="5" class="muted">no ${esc(status)} cases</td></tr>`}</table></div>`;
  });

  // ================================================================= corpus (admin) ========
  A.register("corpus", async (main) => {
    if (!state.principal || !state.principal.roles.includes("admin")) { main.innerHTML = '<div class="card err">Admin role required.</div>'; return; }
    const s = await api("GET", "/api/v1/corpus");
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Corpus Indexes</h2><p>PubMed, bioRxiv/medRxiv and deck bundles: fetch, verify, activate, update incrementally. Full rebuilds run on the build host.</p></div></div>
      <div class="card"><div class="row"><label class="check"><input type="checkbox" id="enabled" ${s.indexed_enabled ? "checked" : ""}> indexed corpus enabled (off = answer from uploaded knowledge bases only)</label></div></div>
      <div class="card"><h3>Bundles on disk</h3><table class="table"><tr><th>corpus</th><th>created</th><th>documents / passages</th><th>encoder</th><th>dense</th><th>updates</th><th>size</th><th></th></tr>
      ${s.bundles.map((b) => `<tr><td><b>${esc(b.corpus_id)}</b> ${b.active ? chip("active", "ok") : ""}${b.encoder_matches_profile ? "" : chip("encoder ≠ profile", "bad")}</td><td class="small">${esc(b.created_at)}</td><td>${b.counts.documents ?? "?"} / ${b.counts.passages ?? "?"}</td><td class="mono">${esc(b.encoder)}</td><td class="small">${esc(b.dense.type || "")} n=${b.dense.n ?? "?"}${b.dense.stale_vectors ? ` stale=${b.dense.stale_vectors}` : ""}</td><td>${b.updates.length}</td><td>${b.total_mb} MB</td><td class="row"><button class="btn ghost small" data-job="verify" data-id="${esc(b.corpus_id)}">verify</button>${b.active ? "" : `<button class="btn small" data-act="${esc(b.corpus_id)}">activate</button>`}</td></tr>`).join("") || '<tr><td colspan="8" class="muted">no bundles in the artifacts directory</td></tr>'}</table></div>
      <div class="two">
        <div class="card"><h3>Fetch a bundle</h3><p class="small muted">From a local path, a URL, or a Google Drive folder synced to the host (rclone).</p><input class="inp" id="f-src" type="text" placeholder="source path or https://…" style="width:100%"><input class="inp" id="f-id" type="text" placeholder="corpus id (folder name)" style="width:100%;margin-top:6px"><div class="row" style="margin-top:6px"><button class="btn" data-run="fetch">Fetch + verify</button></div></div>
        <div class="card"><h3>Update with new documents</h3><p class="small muted">Append parsed JSONL (PubMed daily update files via <span class="mono">scripts.corpus.pubmed_baseline parse</span>, or a preprint window below).</p>
          <select class="inp" id="u-id">${s.bundles.map((b) => `<option ${b.active ? "selected" : ""}>${esc(b.corpus_id)}</option>`).join("")}</select><input class="inp" id="u-docs" type="text" placeholder="JSONL glob(s) on the host, comma-separated" style="width:100%;margin-top:6px"><input class="inp" id="u-enc" type="text" placeholder="encoder spec (must match the bundle)" style="width:100%;margin-top:6px"><div class="row" style="margin-top:6px"><button class="btn" data-run="update">Update index</button></div>
          <hr style="border-color:var(--line)"><div class="row"><select class="inp" id="p-server"><option>biorxiv</option><option>medrxiv</option></select><input class="inp" id="p-from" type="text" placeholder="from YYYY-MM-DD"><input class="inp" id="p-to" type="text" placeholder="to YYYY-MM-DD"><button class="btn" data-run="fetch_preprints">Fetch preprints + update</button></div></div>
      </div>
      <div class="card"><h3>Full rebuild</h3><p class="small muted">Refused on the serving host. Run on the build host, then fetch the bundle here:</p><pre class="log">${esc(s.rebuild_hint)}</pre></div>
      <div class="card"><h3>Jobs</h3><table class="table" id="jobtable"><tr><th>job</th><th>type</th><th>status</th><th>result / error</th><th></th></tr>${s.jobs.map((j) => jobRow(j)).join("") || '<tr><td colspan="5" class="muted">none</td></tr>'}</table><pre class="log" id="joblog" hidden></pre></div>`;
    function jobRow(j) { return `<tr><td class="mono">${esc(j.job_id)}</td><td>${esc(j.type)}</td><td>${chip(j.status, j.status === "done" ? "ok" : j.status === "failed" ? "bad" : "warn")}</td><td class="small">${esc(j.error || JSON.stringify(j.result || {}))}</td><td><button class="btn ghost small" data-log="${esc(j.job_id)}">log</button></td></tr>`; }
    const run = async (type, params) => { try { const j = await api("POST", "/api/v1/corpus/jobs", { type, params }); toast(`job ${j.job_id} started`); setTimeout(A.route, 1200); } catch (e) { toast(e.message, 8000); } };
    main.addEventListener("click", async (e) => {
      const b = e.target.closest("button"); if (!b) return;
      if (b.dataset.job) run(b.dataset.job, { corpus_id: b.dataset.id });
      if (b.dataset.act) { try { await api("POST", "/api/v1/corpus/activate", { corpus_id: b.dataset.act }); toast("activated"); A.route(); } catch (err) { toast(err.message, 8000); } }
      if (b.dataset.run === "fetch") run("fetch", { source: $("f-src").value, corpus_id: $("f-id").value });
      if (b.dataset.run === "update") run("update", { corpus_id: $("u-id").value, docs: $("u-docs").value.split(",").map((x) => x.trim()).filter(Boolean), encoder: $("u-enc").value || undefined });
      if (b.dataset.run === "fetch_preprints") run("fetch_preprints", { server: $("p-server").value, from: $("p-from").value, to: $("p-to").value, corpus_id: $("u-id").value, encoder: $("u-enc").value || undefined });
      if (b.dataset.log) { try { const j = await api("GET", `/api/v1/corpus/jobs/${b.dataset.log}`); $("joblog").hidden = false; $("joblog").textContent = j.log_tail.join("\n"); } catch (err) { toast(err.message); } }
    });
    $("enabled").addEventListener("change", async (e) => { try { await api("POST", "/api/v1/corpus/enabled", { enabled: e.target.checked }); toast(e.target.checked ? "indexed corpus enabled" : "indexed corpus disabled: uploads only"); } catch (err) { toast(err.message); } });
  });

  // ================================================================= trust =================
  A.register("trust", async (main) => {
    const [h, m, x] = await Promise.all([api("GET", "/health"), api("GET", "/api/v1/models"), api("GET", "/api/v1/metrics")]);
    const models = Object.fromEntries(Object.entries(m.models || {}).map(([r, s]) => [r, `${s.name}${s.calibrated ? " (calibrated)" : ""}`]));
    main.innerHTML = `<div class="hero"><img src="/static/logo.svg" width="36"><div><h2>Trust &amp; Versions</h2><p>What this system does, what is in service, what has been measured, and what has not.</p></div></div>
      <div class="card"><h3>What this system does</h3><p>It retrieves passages from the indexed corpus and your uploaded material, writes an answer as separate claims with citations, verifies every claim against the cited passage (support, contradiction, entities, numbers, citation resolution), checks that the answer is about the entities you asked about, and then decides whether to deliver it: <b>ANSWER</b>, <b>ANSWER WITH CAUTION</b>, <b>CLARIFY</b>, <b>ABSTAIN</b> or <b>ESCALATE</b>. Withheld answers go to a human reviewer. It is not medical advice and does not answer patient-specific or dosing questions.</p></div>
      <div class="two"><div class="card"><h3>Versions in service</h3>${kv({ status: h.status, profile: m.profile, corpus_bundle: `${m.bundle_id} ${m.bundle_created_at || ""}`, prompt: m.prompt_id, gate_calibrated_on: m.gate_calibrated_on, api: m.api_version, gpu_preflight: h.preflight.fits ? `fits (${h.preflight.peak_gb} GB peak of ${h.preflight.vram_gb} GB)` : "does not fit", ...models })}</div>
      <div class="card"><h3>Measured performance</h3>${x.status === "NOT YET MEASURED" ? "<b>NOT YET MEASURED.</b> No benchmark run has been recorded for this bundle and profile." : kv(x)}<p class="small muted">Numbers come from benchmark runs on held-out splits and adversarial suites. If a number is missing it has not been measured and no claim is made about it.</p></div></div>
      <div class="card"><h3>Known limits</h3><ul><li>Confidence is shown as <i>uncalibrated</i> until the calibration job has run for the served verifier.</li><li>Confusable-entity detection covers numbered families, prefixes, drug classes and a curated look-alike list; unrelated names not in that list are not caught.</li><li>Evidence quality is shown by tier (peer-reviewed, preprint, course material, uploaded); uploaded material has whatever provenance its owner gave it.</li><li>Human expert labels are collected from in-product feedback and reviewer decisions; until enough exist, reported metrics use proxy labels.</li></ul></div>`;
  });
})(window.AEOS);
