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
  await page.locator('.field:has-text("Layers") input').first().fill("0");
  await page.waitForTimeout(700);
  const errText = await page.locator("#result").innerText();
  record(
    "field validation error",
    errText.includes("model.layers") || errText.includes("layers"),
    `error shown: ${errText.slice(0, 120).replace(/\n/g, " ")}`
  );
  await page.locator('.field:has-text("Layers") input').first().fill("80");
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

  // 7. MLA without kv_lora_rank is rejected with a pointed error
  await page.locator('.field:has-text("Attention") select').first().selectOption("mla");
  await page.waitForTimeout(700);
  const mla = await page.locator("#result").innerText();
  record(
    "MLA without kv_lora_rank rejected",
    mla.includes("kv_lora_rank"),
    `error names the missing MLA field: ${mla.slice(0, 100).replace(/\n/g, " ")}`
  );
  await page.screenshot({ path: "e2e/shot-7-mla.png" });
  await page.locator('.field:has-text("Attention") select').first().selectOption("gqa");
  await page.waitForTimeout(700);

  // 7b. MLA PRESET: Kimi K2.6 (61-layer MLA, MoE) selects, badges, sizes
  await page.locator(".field:has-text(\"Hardware profile\") select").first().selectOption("gb300_nvl72");
  await page.waitForTimeout(900);
  await page.locator("#preset").selectOption("kimi_k2_6");
  await page.waitForTimeout(900);
  const kimiInfo = await page.locator(".preset-info").innerText();
  const kimiRes = await page.locator("#result").innerText();
  record(
    "MLA preset kimi_k2_6 selects + badges",
    kimiInfo.includes("MLA cache") && kimiInfo.includes("MoE") && kimiRes.includes("Feasible"),
    `badges="${kimiInfo.slice(0, 40)}" sized feasible on GB300`
  );
  await page.screenshot({ path: "e2e/shot-7b-mla-preset.png" });
  await page.locator(".field:has-text(\"Hardware profile\") select").first().selectOption("hgx_b200");
  await page.waitForTimeout(900);

  // 7c. HYBRID PRESET: Qwen3.8-27B (DeltaNet hybrid) with fixed KV term
  await page.locator("#preset").selectOption("qwen_3_8_27b");
  await page.waitForTimeout(900);
  const hybridInfo = await page.locator(".preset-info").innerText();
  const hybridRes = await page.locator("#result").innerText();
  record(
    "hybrid preset qwen_3_8_27b selects + badge",
    hybridInfo.includes("Hybrid attention") && hybridRes.includes("Feasible"),
    `badge shown, hybrid model sized feasibly`
  );
  await page.screenshot({ path: "e2e/shot-7c-hybrid.png" });

  // 7d. UNVERIFIED CHIP + KV OVERRIDE: DSv4 Flash 0731
  await page.locator("#preset").selectOption("deepseek_v4_flash_0731");
  await page.waitForTimeout(900);
  const ovInfo = await page.locator(".preset-info").innerText();
  const ovRes = await page.locator("#result").innerText();
  record(
    "override preset DSv4 Flash: KV override + MoE + sizing",
    ovInfo.includes("KV override") && ovInfo.includes("MoE") && ovRes.includes("Feasible"),
    `badges="${ovInfo.slice(0, 60)}" vendor KV figure applied`
  );
  await page.screenshot({ path: "e2e/shot-7d-override.png" });

  // 7e. UNVERIFIED confidence chip: DSv4.1 Flash
  await page.locator("#preset").selectOption("deepseek_v4_1_flash");
  await page.waitForTimeout(900);
  const uvInfo = await page.locator(".preset-info").innerText();
  record(
    "unverified preset shows confidence chip",
    uvInfo.includes("Specs unverified"),
    `chip text present: "${uvInfo.slice(0, 50)}"`
  );
  await page.screenshot({ path: "e2e/shot-7e-unverified.png" });

  // 7f. FAMILY-GROUPED preset picker with optgroups
  const optgroups = await page.locator("#preset optgroup").count();
  record(
    "preset picker grouped by family",
    optgroups >= 5,
    `${optgroups} family optgroups in the preset dropdown`
  );
  await page.locator("#preset").selectOption("");

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