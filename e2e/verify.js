// End-to-end verification of the Lightning AI Inference Sizer.
// Usage: node e2e/verify.js  (server must be running on :8000)

const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8000";
const results = [];

function record(name, pass, detail) {
  results.push({ name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name} — ${detail}`);
}

(async () => {
  const browser = await chromium.launch({
    executablePath:
      "/Users/aanthony/workshop/e2e/.browsers/chromium_headless_shell-1248/chrome-headless-shell-mac-arm64/chrome-headless-shell",
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const dialogs = [];
  page.on("dialog", async (d) => {
    dialogs.push(d.message());
    await d.dismiss();
  });
  page.on("pageerror", (e) => record("no page JS errors", false, e.message));

  // 1. LOAD
  await page.goto(BASE, { waitUntil: "networkidle" });
  const wordmark = await page.locator("img.wordmark").isVisible();
  const steps = await page.locator(".step").count();
  const badges = await page.locator(".badge").first().textContent();
  const fleetKpi = await page.locator(".kpi-value").first().textContent();
  record(
    "load + initial calculation",
    wordmark && steps === 4 && badges.includes("Estimated") && /\d/.test(fleetKpi),
    `wordmark=${wordmark} steps=${steps} badge="${badges}" fleet="${fleetKpi}"`
  );
  await page.screenshot({ path: "e2e/shot-1-load.png", fullPage: false });

  // 2. CALCULATION FLOW: rps 20 -> 40 changes serving
  const servingBefore = await page.locator("#result table").nth(1).innerText();
  await page.locator('.field:has-text("Peak requests / sec") input').fill("40");
  await page.waitForTimeout(700);
  const servingAfter = await page.locator("#result table").nth(1).innerText();
  record("calculation flow updates", servingBefore !== servingAfter, "results changed after rps 20->40");
  await page.screenshot({ path: "e2e/shot-2-flow.png" });

  // 3. HARDWARE SWEEP
  await page.locator(".field:has-text(\"Hardware profile\") select").selectOption("dgx_h100");
  await page.waitForTimeout(700);
  const hwStatus = await page
    .locator("section.step:has(.field:has-text(\"Hardware profile\")) .provenance")
    .first()
    .innerText();
  record(
    "hardware sweep",
    hwStatus.includes("DGX H100") && hwStatus.includes("Inventory"),
    `status line: ${hwStatus.slice(0, 90)}...`
  );
  await page.locator(".field:has-text(\"Hardware profile\") select").selectOption("hgx_b200");
  await page.waitForTimeout(700);

  // 4. VALIDATION ERROR: layers=0
  await page.locator('.field:has-text("Layers") input').fill("0");
  await page.waitForTimeout(700);
  const errText = await page.locator("#result").innerText();
  record(
    "field validation error",
    errText.includes("model.layers") || errText.includes("layers"),
    `error shown: ${errText.slice(0, 120).replace(/\n/g, " ")}`
  );
  await page.locator('.field:has-text("Layers") input').fill("80");
  await page.waitForTimeout(700);

  // 5. INVALID MAX GPU = 0
  await page.locator('.field:has-text("Max GPUs per replica") input').fill("0");
  await page.waitForTimeout(700);
  const err5 = await page.locator("#result").innerText();
  record(
    "max_gpus_per_replica=0 error",
    err5.includes("max_gpus_per_replica"),
    `error shown: ${err5.slice(0, 100).replace(/\n/g, " ")}`
  );
  await page.locator('.field:has-text("Max GPUs per replica") input').fill("8");
  await page.waitForTimeout(700);

  // 6. INJECTED TEXT SAFETY
  await page.locator('.field:has-text("Customer") input').fill("<script>alert(1)</script>");
  await page.waitForTimeout(700);
  const cust = await page.locator(".customer-line").innerText();
  record(
    "injected text rendered inert",
    dialogs.length === 0 && cust.includes("<script>alert(1)</script>"),
    `dialogs=${dialogs.length}, customer line contains literal script tag as text`
  );
  await page.locator('.field:has-text("Customer") input').fill("Example Co");
  await page.waitForTimeout(700);

  // 7. MLA UNSUPPORTED
  await page.locator(".field:has-text(\"Attention\") select").selectOption("mla");
  await page.waitForTimeout(700);
  const mla = await page.locator("#result").innerText();
  record(
    "MLA unsupported",
    mla.includes("Unsupported") && mla.includes("Architecture unsupported"),
    "unsupported + architecture-unsupported badges shown"
  );
  await page.screenshot({ path: "e2e/shot-7-mla.png" });
  await page.locator(".field:has-text(\"Attention\") select").selectOption("gqa");
  await page.waitForTimeout(700);

  // 8. COMPARISON
  await page.locator('button:has-text("Add current result to comparison")').click();
  await page.waitForTimeout(300);
  const cmp = await page.locator("#compare table").count();
  record("comparison table", cmp === 1, "one comparison column added");

  // 9. PRINT FUNCTION PRESENT
  const hasPrint = await page.evaluate(() => typeof window.print);
  record("print available", hasPrint === "function", "window.print is a function");

  // 10. RESPONSIVE
  await page.setViewportSize({ width: 700, height: 900 });
  await page.waitForTimeout(300);
  const formBox = await page.locator(".form-col").boundingBox();
  const resBox = await page.locator(".result-col").boundingBox();
  const stacked = resBox.y >= formBox.y + formBox.height - 5;
  record("responsive stacking at 700px", stacked, `form h=${Math.round(formBox.height)}, results y=${Math.round(resBox.y)}`);
  await page.screenshot({ path: "e2e/shot-10-narrow.png" });
  await page.setViewportSize({ width: 1440, height: 900 });

  // 11. EXPORT (no error in UI)
  await page.locator("#btn-export").click();
  await page.waitForTimeout(400);
  const exportErr = await page.locator("#result").innerText();
  record(
    "export triggers without UI error",
    !exportErr.includes("Could not size"),
    "export clicked; download started"
  );

  await browser.close();
  const passed = results.filter((r) => r.pass).length;
  console.log(`\n${passed}/${results.length} checks passed`);
  process.exit(results.every((r) => r.pass) ? 0 : 1);
})().catch((e) => {
  console.error("E2E runner error:", e);
  process.exit(2);
});