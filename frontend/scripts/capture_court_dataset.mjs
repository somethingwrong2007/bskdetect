import { chromium } from 'playwright';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

function arg(name, fallback = '') {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

const specsPath = path.resolve(arg('specs'));
const outDir = path.resolve(arg('out'));
const baseUrl = arg('url', 'http://127.0.0.1:5173');
const width = Number(arg('width', '960'));
const height = Number(arg('height', '540'));
const humans = process.argv.includes('--humans');
const simpleHumans = process.argv.includes('--simple-humans');
const resume = process.argv.includes('--resume');
const start = Number(arg('start', '0'));
const limit = Number(arg('limit', 'Infinity'));
const batchSize = Math.max(1, Number(arg('batch-size', '8')));
const specs = (await readFile(specsPath, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
const metaPath = path.join(outDir, 'meta.jsonl');
const existing = [];
if (resume) {
  try {
    const text = await readFile(metaPath, 'utf8');
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      try {
        existing.push(JSON.parse(line));
      } catch {
        console.warn('resume: 跳过未写完的 meta 行');
      }
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
}
const done = new Set(existing.map((item) => item.idx));
const selected = specs
  .slice(start, limit === Infinity ? undefined : start + limit)
  .filter((spec) => !done.has(spec.idx));
const groups = new Map();
for (const spec of selected) {
  if (!groups.has(spec.team)) groups.set(spec.team, []);
  groups.get(spec.team).push(spec);
}

await mkdir(path.join(outDir, 'images'), { recursive: true });
await mkdir(path.join(outDir, 'labels'), { recursive: true });
if (!resume) await writeFile(metaPath, '');

const browser = await chromium.launch({
  headless: false,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist', '--disable-gpu-sandbox'],
});
let completed = 0;
try {
  for (const [team, teamSpecs] of groups) {
    const page = await browser.newPage({ viewport: { width, height } });
    page.on('pageerror', (error) => console.error(`pageerror: ${error.stack}`));
    page.on('console', (message) => {
      if (message.type() === 'error') console.error(`console: ${message.text()}`);
    });
    const query = new URLSearchParams({
      team,
      ...(humans ? { humans: '1' } : {}),
      ...(simpleHumans ? { simplehumans: '1' } : {}),
    });
    for (let attempt = 0; attempt < 40; attempt++) {
      try {
        await page.goto(`${baseUrl}/?${query}`, { waitUntil: 'networkidle', timeout: 120000 });
        break;
      } catch (error) {
        if (attempt === 39) throw error;
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }
    await page.waitForFunction(() => window.__ready, { timeout: 120000 });
    const error = await page.evaluate(() => window.__error || null);
    if (error) throw new Error(error);
    for (let offset = 0; offset < teamSpecs.length; offset += batchSize) {
      const batch = teamSpecs.slice(offset, offset + batchSize);
      const samples = await page.evaluate(
        (cameras) => window.captureDatasetBatch(cameras),
        batch.map((spec) => spec.camera),
      );
      if (samples.length !== batch.length) {
        throw new Error(`batch returned ${samples.length}/${batch.length} samples`);
      }
      const lines = [];
      await Promise.all(batch.map(async (spec, i) => {
        const sample = samples[i];
        if (sample.width !== width || sample.height !== height) {
          throw new Error(`unexpected capture size ${sample.width}x${sample.height}`);
        }
        const imagePath = path.join(outDir, 'images', `${String(spec.idx).padStart(6, '0')}.jpg`);
        const labelPath = path.join(outDir, 'labels', `${String(spec.idx).padStart(6, '0')}.png`);
        await Promise.all([
          writeFile(imagePath, Buffer.from(sample.image.split(',')[1], 'base64')),
          writeFile(labelPath, Buffer.from(sample.label.split(',')[1], 'base64')),
        ]);
        lines.push(JSON.stringify({
          ...spec,
          H_world_to_px: sample.H_world_to_px,
          image_size: [height, width],
          intrinsics: { fx: spec.camera.focal_length, fy: spec.camera.focal_length,
            cx: width / 2, cy: height / 2, distortion: [0, 0, 0, 0, 0] },
          renderer: 'three-webgpu-r170',
          humans,
          simple_humans: simpleHumans,
        }) + '\n');
      }));
      await appendFile(metaPath, lines.join(''));
      completed += batch.length;
      if (completed % 50 === 0) console.log(`captured ${completed}/${selected.length}`);
    }
    await page.close();
  }
} finally {
  await browser.close();
}
console.log(`saved ${completed} WebGPU samples to ${outDir}`);
