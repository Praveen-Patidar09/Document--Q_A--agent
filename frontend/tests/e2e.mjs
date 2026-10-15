import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";

const vite = await createServer({
  configFile: "vite.config.js",
  server: { host: "127.0.0.1", port: 5187, strictPort: true },
});
let browser;
const docs = [];
const calls = { deletes: [], questions: [] };
const firstId = "111111111111111111111111";
const secondId = "222222222222222222222222";

try {
  await vite.listen();
  browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.route("**/health", (route) => route.fulfill({ json: { status: "ok" } }));
  await page.route("**/documents", async (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({ json: { documents: [...docs] } });
    }
    const body = route.request().postDataBuffer()?.toString("latin1") || "";
    if (body.includes("blocked.xyz")) {
      return route.fulfill({ status: 415, json: { detail: "Unsupported file type." } });
    }
    const second = body.includes("work.csv");
    const record = { document_id: second ? secondId : firstId,
      filename: second ? "work.csv" : "notes.pdf", chunks: 3,
      created_at: "2026-10-06T12:00:00Z" };
    const existing = docs.findIndex((doc) => doc.document_id === record.document_id);
    if (existing >= 0) docs.splice(existing, 1);
    docs.unshift(record);
    return route.fulfill({ status: 201, json: {
      document_id: record.document_id, filename: record.filename, chunks: record.chunks,
    } });
  });
  await page.route("**/documents/*", async (route) => {
    const id = route.request().url().split("/").at(-1);
    calls.deletes.push(id);
    const index = docs.findIndex((doc) => doc.document_id === id);
    if (index < 0) return route.fulfill({ status: 404, json: { detail: "Document not found." } });
    docs.splice(index, 1);
    return route.fulfill({ status: 204, body: "" });
  });
  await page.route("**/questions", async (route) => {
    const payload = JSON.parse(route.request().postData() || "{}");
    calls.questions.push(payload);
    if (payload.question === "trigger backend error") {
      return route.fulfill({ status: 503, json: { detail: "Question answering is temporarily unavailable." } });
    }
    const filename = docs.find((doc) => doc.document_id === payload.document_id)?.filename;
    return route.fulfill({ json: { answer: `Answer for ${filename}.`,
      sources: [{ filename: "notes.pdf", page: 2, chunk: 0, similarity: 0.91 }] } });
  });

  await page.goto("http://127.0.0.1:5187");
  await page.getByText("Your library is empty").waitFor();
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles({ name: "notes.pdf", mimeType: "application/pdf", buffer: Buffer.from("sample") });
  await page.locator(".document-select").filter({ hasText: "notes.pdf" }).waitFor();
  await page.getByLabel("YOUR QUESTION").fill("What is in my notes?");
  await page.getByRole("button", { name: /Ask question/ }).click();
  await page.getByText("Answer for notes.pdf.").waitFor();
  await page.getByText(/notes\.pdf · page 2 · chunk 1/).waitFor();
  assert.equal(calls.questions.at(-1).document_id, firstId, "question must include selected document ID");

  await fileInput.setInputFiles({ name: "work.csv", mimeType: "text/csv", buffer: Buffer.from("name,value\nA,1") });
  await page.locator(".document-select").filter({ hasText: "work.csv" }).waitFor();
  await page.locator(".document-select").filter({ hasText: "notes.pdf" }).click();
  await page.getByLabel("YOUR QUESTION").fill("Confirm selected source");
  await page.getByRole("button", { name: /Ask question/ }).click();
  await page.getByText("Answer for notes.pdf.").waitFor();
  assert.equal(calls.questions.at(-1).document_id, firstId, "query must stay in the selected document");

  await page.getByRole("button", { name: "Clear" }).click();
  assert.equal(await page.getByLabel("YOUR QUESTION").inputValue(), "");
  await assert.rejects(() => page.getByText("Answer for notes.pdf.").waitFor({ timeout: 100 }));

  await fileInput.setInputFiles({ name: "blocked.xyz", mimeType: "application/octet-stream", buffer: Buffer.from("bad") });
  await page.getByRole("alert").getByText("Unsupported file type.").waitFor();
  await page.getByLabel("YOUR QUESTION").fill("trigger backend error");
  await page.getByRole("button", { name: /Ask question/ }).click();
  await page.getByRole("alert").getByText("Question answering is temporarily unavailable.").waitFor();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Delete notes.pdf" }).click();
  await page.locator(".document-row").filter({ hasText: "notes.pdf" }).waitFor({ state: "detached" });
  assert.deepEqual(calls.deletes, [firstId]);
  assert.equal(await page.locator(".document-row").count(), 1, "other document must remain in the library");
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    "mobile layout should not overflow horizontally");
  assert.deepEqual(pageErrors, [], "frontend should not throw browser errors");
  console.log("PASS frontend upload, selection, scoped Q&A, citations, clear, errors, and delete");
} finally {
  if (browser) await browser.close();
  await vite.close();
}
