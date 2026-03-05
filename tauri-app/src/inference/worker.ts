import * as ort from "onnxruntime-web/webgpu";

function post(msg: unknown) {
  self.postMessage(msg);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];
const YOLO_SIZE = 640;
const BBOX_PADDING = 0.15;
const YOLO_BIRD_CLASS = 14; // bird in YOLO's 80-class COCO indexing

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let detector: ort.InferenceSession | null = null;
let classifier: ort.InferenceSession | null = null;
let idxToName = new Map<number, string>();

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      await handleInit(msg);
      break;
    case "process":
      await handleProcess(msg);
      break;
    case "dispose":
      handleDispose();
      break;
  }
};

// ---------------------------------------------------------------------------
// Init — load models
// ---------------------------------------------------------------------------

async function handleInit(msg: {
  detectorUrl: string;
  classifierUrl: string;
  labelMapUrl: string;
}) {
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

    // Load detector with WebGPU (YOLOv8n — fixed 640x640 input)
    post({ type: "init-status", status: "Loading detector model..." });
    const detectorBytes = await fetch(msg.detectorUrl).then((r) =>
      r.arrayBuffer(),
    );
    try {
      detector = await ort.InferenceSession.create(detectorBytes, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("[ort-web worker] detector loaded with WebGPU");
    } catch {
      detector = await ort.InferenceSession.create(detectorBytes, {
        executionProviders: ["wasm"],
      });
      console.log("[ort-web worker] detector loaded with WASM fallback");
    }

    // Load classifier with WebGPU (fixed 224x224 input)
    post({ type: "init-status", status: "Loading classifier model..." });
    const classifierBytes = await fetch(msg.classifierUrl).then((r) =>
      r.arrayBuffer(),
    );
    try {
      classifier = await ort.InferenceSession.create(classifierBytes, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("[ort-web worker] classifier loaded with WebGPU");
    } catch {
      classifier = await ort.InferenceSession.create(classifierBytes, {
        executionProviders: ["wasm"],
      });
      console.log("[ort-web worker] classifier loaded with WASM fallback");
    }

    console.log(
      `[worker] Models loaded in ${((performance.now() - initStart) / 1000).toFixed(1)}s`,
    );
    post({ type: "init-done" });
  } catch (err) {
    post({ type: "init-error", error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Process — detect + classify a single image
// ---------------------------------------------------------------------------

async function handleProcess(msg: {
  url: string;
  path: string;
  folder: string;
  fileMtime: number;
  fileSize: number;
  detectThreshold: number;
  minConfidence: number;
  topK: number;
}) {
  try {
    if (!detector || !classifier) {
      post({ type: "error", path: msg.path, error: "Models not loaded" });
      return;
    }

    const t0 = performance.now();

    // Load image via fetch + createImageBitmap (no DOM needed)
    const response = await fetch(msg.url);
    const blob = await response.blob();
    const img = await createImageBitmap(blob);
    const tLoad = performance.now();

    // --- Detection (YOLOv8) ---
    const det = detectorPreprocess(img);
    const tDetPrep = performance.now();

    const detResults = await detector.run({ images: det.tensor });
    det.tensor.dispose();
    const tDetRun = performance.now();

    // YOLOv8 output: [1, 4+numClasses, numDetections]
    const output0 = detResults["output0"];
    const data = output0.data as Float32Array;
    const [, , numDetections] = output0.dims;
    output0.dispose();

    let bestBbox: [number, number, number, number] | null = null;
    let bestScore = 0;

    for (let i = 0; i < numDetections; i++) {
      // Bird class score (class 14, offset by 4 bbox channels)
      const birdScore =
        data[(4 + YOLO_BIRD_CLASS) * numDetections + i];

      if (birdScore >= msg.detectThreshold && birdScore > bestScore) {
        // Extract cx, cy, w, h in 640x640 letterboxed coords
        const cx = data[0 * numDetections + i];
        const cy = data[1 * numDetections + i];
        const w = data[2 * numDetections + i];
        const h = data[3 * numDetections + i];

        // Convert to x1,y1,x2,y2 and undo letterbox → original image coords
        const x1 = Math.max(0, (cx - w / 2 - det.padX) / det.scale);
        const y1 = Math.max(0, (cy - h / 2 - det.padY) / det.scale);
        const x2 = Math.min(img.width, (cx + w / 2 - det.padX) / det.scale);
        const y2 = Math.min(img.height, (cy + h / 2 - det.padY) / det.scale);

        bestBbox = [x1, y1, x2, y2];
        bestScore = birdScore;
      }
    }

    // Also check if any other class scored higher on the same detections
    // (skip if no bird found at all)
    if (!bestBbox) {
      img.close();
      console.log(
        `[worker] ${msg.path.split("/").pop()} — SKIP | load=${(tLoad - t0).toFixed(0)}ms det-prep=${(tDetPrep - tLoad).toFixed(0)}ms det-run=${(tDetRun - tDetPrep).toFixed(0)}ms total=${(performance.now() - t0).toFixed(0)}ms`,
      );
      post({
        type: "result",
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

    // --- Classification ---
    const input = cropAndPreprocess(img, bestBbox);
    img.close();
    const tClsPrep = performance.now();

    const classResults = await classifier.run({ image: input });
    input.dispose();
    const tClsRun = performance.now();

    const logits = classResults.logits.data as Float32Array;
    classResults.logits.dispose();

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
        type: "result",
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
    console.log(
      `[worker] ${msg.path.split("/").pop()} — ${species} (${(top1.confidence * 100).toFixed(1)}%) | load=${(tLoad - t0).toFixed(0)}ms det-prep=${(tDetPrep - tLoad).toFixed(0)}ms det-run=${(tDetRun - tDetPrep).toFixed(0)}ms cls-prep=${(tClsPrep - tDetRun).toFixed(0)}ms cls-run=${(tClsRun - tClsPrep).toFixed(0)}ms post=${(tEnd - tClsRun).toFixed(0)}ms total=${(tEnd - t0).toFixed(0)}ms`,
    );

    post({
      type: "result",
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
    post({ type: "error", path: msg.path, error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Dispose
// ---------------------------------------------------------------------------

function handleDispose() {
  detector?.release();
  classifier?.release();
  detector = null;
  classifier = null;
}

// ---------------------------------------------------------------------------
// Preprocessing (uses OffscreenCanvas + ImageBitmap — no DOM needed)
// ---------------------------------------------------------------------------

/** Letterbox image to 640x640 for YOLOv8. */
function detectorPreprocess(img: ImageBitmap): {
  tensor: ort.Tensor;
  scale: number;
  padX: number;
  padY: number;
} {
  const origW = img.width;
  const origH = img.height;

  // Fit in 640x640 maintaining aspect ratio
  const scale = Math.min(YOLO_SIZE / origW, YOLO_SIZE / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padX = (YOLO_SIZE - newW) / 2;
  const padY = (YOLO_SIZE - newH) / 2;

  const canvas = new OffscreenCanvas(YOLO_SIZE, YOLO_SIZE);
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(114,114,114)"; // YOLO standard padding color
  ctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  ctx.drawImage(img, padX, padY, newW, newH);

  const imageData = ctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);
  const pixels = YOLO_SIZE * YOLO_SIZE;
  const float32 = new Float32Array(3 * pixels);
  const rgba = imageData.data;
  for (let i = 0; i < pixels; i++) {
    const base = i * 4;
    float32[i] = rgba[base] / 255;
    float32[pixels + i] = rgba[base + 1] / 255;
    float32[2 * pixels + i] = rgba[base + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", float32, [1, 3, YOLO_SIZE, YOLO_SIZE]),
    scale,
    padX,
    padY,
  };
}

/** Crop bbox with padding → pad to square → resize 224 → CLIP normalize. */
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
  const scale = 224 / side;

  const canvas = new OffscreenCanvas(224, 224);
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(128,128,128)";
  ctx.fillRect(0, 0, 224, 224);
  ctx.drawImage(
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

  const imageData = ctx.getImageData(0, 0, 224, 224);
  return rgbaToClipTensor(imageData);
}

function rgbaToClipTensor(imageData: ImageData): ort.Tensor {
  const pixels = imageData.width * imageData.height;
  const float32 = new Float32Array(3 * pixels);
  const rgba = imageData.data;
  for (let i = 0; i < pixels; i++) {
    const base = i * 4;
    float32[i] = (rgba[base] / 255 - CLIP_MEAN[0]) / CLIP_STD[0];
    float32[pixels + i] =
      (rgba[base + 1] / 255 - CLIP_MEAN[1]) / CLIP_STD[1];
    float32[2 * pixels + i] =
      (rgba[base + 2] / 255 - CLIP_MEAN[2]) / CLIP_STD[2];
  }
  return new ort.Tensor("float32", float32, [
    1,
    3,
    imageData.height,
    imageData.width,
  ]);
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
