import * as ort from "onnxruntime-web/webgpu";

function post(msg: unknown) {
  self.postMessage(msg);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];
const BBOX_PADDING = 0.15;
const TEMPERATURE = 0.74; // calibrated via calibrate.py — sharpens underconfident logits

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let classifier: ort.InferenceSession | null = null;
let idxToName = new Map<number, string>();

// Pre-allocated buffer for classifier preprocessing (reused across images)
const CLS_SIZE = 224;
const CLS_PIXELS = CLS_SIZE * CLS_SIZE;
const clsFloat32 = new Float32Array(3 * CLS_PIXELS);
const clsCanvas = new OffscreenCanvas(CLS_SIZE, CLS_SIZE);
const clsCtx = clsCanvas.getContext("2d")!;

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      await handleInit(msg);
      break;
    case "classify":
      await handleClassify(msg);
      break;
    case "dispose":
      handleDispose();
      break;
  }
};

// ---------------------------------------------------------------------------
// Init — load classifier model + label map
// ---------------------------------------------------------------------------

async function handleInit(msg: { classifierUrl: string; labelMapUrl: string }) {
  try {
    ort.env.wasm.wasmPaths = "/ort-wasm/";
    ort.env.wasm.numThreads = 1;
    const initStart = performance.now();

    post({ type: "init-status", status: "Loading label map..." });
    const labelMap: Record<string, number> = await fetch(msg.labelMapUrl).then(
      (r) => r.json(),
    );
    idxToName = new Map<number, string>();
    for (const [name, idx] of Object.entries(labelMap)) {
      idxToName.set(idx, name);
    }

    post({ type: "init-status", status: "Loading classifier model..." });
    const classifierBytes = await fetch(msg.classifierUrl).then((r) =>
      r.arrayBuffer(),
    );
    try {
      classifier = await ort.InferenceSession.create(classifierBytes, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("[classifier] loaded with WebGPU");
    } catch {
      classifier = await ort.InferenceSession.create(classifierBytes, {
        executionProviders: ["wasm"],
      });
      console.log("[classifier] loaded with WASM fallback");
    }

    console.log(
      `[classifier] Model loaded in ${((performance.now() - initStart) / 1000).toFixed(1)}s`,
    );
    post({ type: "init-done" });
  } catch (err) {
    post({ type: "init-error", error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Classify — crop bbox from image, run classifier, return species + topK
// ---------------------------------------------------------------------------

async function handleClassify(msg: {
  url: string;
  path: string;
  folder: string;
  fileMtime: number;
  fileSize: number;
  bbox: [number, number, number, number];
  minConfidence: number;
  topK: number;
}) {
  try {
    if (!classifier) {
      post({
        type: "classify-error",
        path: msg.path,
        error: "Classifier not loaded",
      });
      return;
    }

    const t0 = performance.now();

    // Load image
    const response = await fetch(msg.url);
    const blob = await response.blob();
    const img = await createImageBitmap(blob);
    const tLoad = performance.now();

    // Crop bbox + preprocess for CLIP
    const input = cropAndPreprocess(img, msg.bbox);
    img.close();
    const tPrep = performance.now();

    const classResults = await classifier.run({ image: input });
    input.dispose();
    const tRun = performance.now();

    const logits = classResults.logits.data as Float32Array;
    classResults.logits.dispose();

    // Temperature scaling: divide logits by T before softmax
    for (let i = 0; i < logits.length; i++) logits[i] /= TEMPERATURE;

    const probs = softmax(logits);

    // Top-K extraction
    const indexed: [number, number][] = [];
    for (let i = 0; i < probs.length; i++) indexed.push([i, probs[i]]);
    indexed.sort((a, b) => b[1] - a[1]);

    const topK = indexed.slice(0, msg.topK).map(([idx, conf]) => ({
      classIdx: idx,
      speciesName: idxToName.get(idx) ?? String(idx),
      confidence: conf,
    }));

    if (topK.length === 0) {
      post({
        type: "classify-result",
        path: msg.path,
        folder: msg.folder,
        species: "__skipped__",
        speciesIdx: -1,
        confidence: 0.0,
        fileMtime: msg.fileMtime,
        fileSize: msg.fileSize,
      });
      return;
    }

    const top1 = topK[0];
    const species =
      top1.confidence < msg.minConfidence ? "__unknown__" : top1.speciesName;

    const topKJson = JSON.stringify(
      topK.map((p) => ({
        scientificName: p.speciesName,
        confidence: Math.round(p.confidence * 10000) / 10000,
      })),
    );

    const tEnd = performance.now();
    const filename = msg.path.split("/").pop() ?? msg.path.split("\\").pop();
    console.log(
      `[classifier] ${filename} — ${species} (${(top1.confidence * 100).toFixed(1)}%) | load=${(tLoad - t0).toFixed(0)}ms prep=${(tPrep - tLoad).toFixed(0)}ms run=${(tRun - tPrep).toFixed(0)}ms total=${(tEnd - t0).toFixed(0)}ms`,
    );

    post({
      type: "classify-result",
      path: msg.path,
      folder: msg.folder,
      species,
      speciesIdx: top1.classIdx,
      confidence: top1.confidence,
      fileMtime: msg.fileMtime,
      fileSize: msg.fileSize,
      topKJson,
    });
  } catch (err) {
    post({ type: "classify-error", path: msg.path, error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Dispose
// ---------------------------------------------------------------------------

function handleDispose() {
  classifier?.release();
  classifier = null;
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/** Crop bbox with padding → pad to square → resize 224 → CLIP normalize. Uses pre-allocated buffers. */
function cropAndPreprocess(
  img: ImageBitmap,
  bbox: [number, number, number, number],
): ort.Tensor {
  const [x1, y1, x2, y2] = bbox;
  const bw = x2 - x1;
  const bh = y2 - y1;

  const cx1 = Math.max(0, x1 - bw * BBOX_PADDING);
  const cy1 = Math.max(0, y1 - bh * BBOX_PADDING);
  const cx2 = Math.min(img.width, x2 + bw * BBOX_PADDING);
  const cy2 = Math.min(img.height, y2 + bh * BBOX_PADDING);
  const cropW = cx2 - cx1;
  const cropH = cy2 - cy1;

  const side = Math.max(cropW, cropH);
  const offsetX = (side - cropW) / 2;
  const offsetY = (side - cropH) / 2;
  const scale = CLS_SIZE / side;

  clsCtx.fillStyle = "rgb(128,128,128)";
  clsCtx.fillRect(0, 0, CLS_SIZE, CLS_SIZE);
  clsCtx.drawImage(
    img,
    cx1,
    cy1,
    cropW,
    cropH,
    offsetX * scale,
    offsetY * scale,
    cropW * scale,
    cropH * scale,
  );

  const imageData = clsCtx.getImageData(0, 0, CLS_SIZE, CLS_SIZE);
  const rgba = imageData.data;
  for (let i = 0; i < CLS_PIXELS; i++) {
    const base = i * 4;
    clsFloat32[i] = (rgba[base] / 255 - CLIP_MEAN[0]) / CLIP_STD[0];
    clsFloat32[CLS_PIXELS + i] = (rgba[base + 1] / 255 - CLIP_MEAN[1]) / CLIP_STD[1];
    clsFloat32[2 * CLS_PIXELS + i] =
      (rgba[base + 2] / 255 - CLIP_MEAN[2]) / CLIP_STD[2];
  }
  return new ort.Tensor("float32", clsFloat32, [1, 3, CLS_SIZE, CLS_SIZE]);
}

function softmax(logits: Float32Array): Float32Array {
  let maxVal = -Infinity;
  for (let i = 0; i < logits.length; i++)
    if (logits[i] > maxVal) maxVal = logits[i];
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - maxVal);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}
