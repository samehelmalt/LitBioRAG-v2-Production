// Screenshots of the running app for the landing page (app/web/static/shots/*.png).
//   node scripts/screenshots.js http://127.0.0.1:8471 <reviewer-jwt> [outDir] [theme]
// Uses the pre-installed Playwright (Chromium). Run by scripts/screenshots.py; the pages are the
// real UI against the demo bundle, so nothing in the pictures is mocked.
const { chromium } = require("playwright");
const path = require("path");

async function main() {
  const [base, reviewerToken, outDir = "app/web/static/shots", theme = "dark"] = process.argv.slice(2);
  if (!base || !reviewerToken) { console.error("usage: node scripts/screenshots.js <base-url> <reviewer-jwt> [outDir] [theme]"); process.exit(2); }
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.addInitScript((t) => { try { localStorage.setItem("aeos.theme", t); } catch (e) {} }, theme);
  const shot = async (name) => { await page.evaluate(() => { const t = document.getElementById("toast"); if (t) t.hidden = true; }); await page.screenshot({ path: path.join(outDir, `${name}.png`), fullPage: false }); };

  // landing (both themes for the review record)
  await page.goto(`${base}/`, { waitUntil: "networkidle" });
  await shot("landing");

  // chat: demo sign-in, ask, wait for the result
  await page.goto(`${base}/app?demo=1#/chat`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => document.getElementById("who") && document.getElementById("who").textContent.startsWith("demo-"), null, { timeout: 15000 });
  await page.fill("#q", "What does MAP3K3 do in NF-kappaB signalling?");  // covered by the demo corpus
  await page.click("#ask button[type=submit]");
  await page.waitForSelector(".bubble .banner", { timeout: 60000 });
  await page.waitForTimeout(400);
  await shot("chat");

  // reviewer pages (a changed query string forces a full reload so the new token is read)
  await page.evaluate((t) => { sessionStorage.setItem("aeos.token", t); }, reviewerToken);
  await page.goto(`${base}/app?as=reviewer#/review`, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);
  await shot("review");
  await page.goto(`${base}/app?as=reviewer2#/upload`, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);
  await shot("upload");
  await page.goto(`${base}/app?as=reviewer3#/trust`, { waitUntil: "networkidle" });
  await page.waitForTimeout(600);
  await shot("trust");
  await browser.close();
  console.log(`screenshots written to ${outDir}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
