import * as ort from "onnxruntime-web/webgpu";

function post(msg: unknown) {
  self.postMessage(msg);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const YOLO_SIZE = 1280;
const YOLO_BIRD_CLASS = 14; // bird in YOLO's 80-class COCO indexing

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let detector: ort.InferenceSession | null = null;

// Pre-allocated buffer for detector preprocessing (reused across images)
const DET_PIXELS = YOLO_SIZE * YOLO_SIZE;
const detFloat32 = new Float32Array(3 * DET_PIXELS);
const detCanvas = new OffscreenCanvas(YOLO_SIZE, YOLO_SIZE);
const detCtx = detCanvas.getContext("2d")!;

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      await handleInit(msg);
      break;
    case "detect":
      await handleDetect(msg);
      break;
    case "dispose":
      handleDispose();
      break;
  }
};

// ---------------------------------------------------------------------------
// Init — load detector model only
// ---------------------------------------------------------------------------

async function handleInit(msg: { detectorUrl: string }) {
  try {
    ort.env.wasm.wasmPaths = "/ort-wasm/";
    ort.env.wasm.numThreads = 1;
    const initStart = performance.now();

    post({ type: "init-status", status: "Loading detector model..." });
    const detectorBytes = await fetch(msg.detectorUrl).then((r) =>
      r.arrayBuffer(),
    );
    try {
      detector = await ort.InferenceSession.create(detectorBytes, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("[detector] loaded with WebGPU");
    } catch {
      detector = await ort.InferenceSession.create(detectorBytes, {
        executionProviders: ["wasm"],
      });
      console.log("[detector] loaded with WASM fallback");
    }

    console.log(
      `[detector] Model loaded in ${((performance.now() - initStart) / 1000).toFixed(1)}s`,
    );
    post({ type: "init-done" });
  } catch (err) {
    post({ type: "init-error", error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Detect — run YOLO on a single image, return best bird bbox
// ---------------------------------------------------------------------------

async function handleDetect(msg: {
  url: string;
  path: string;
  detectThreshold: number;
}) {
  try {
    if (!detector) {
      post({ type: "detect-error", path: msg.path, error: "Detector not loaded" });
      return;
    }

    const t0 = performance.now();

    // Load image + resize to letterbox dimensions via createImageBitmap
    const response = await fetch(msg.url);
    const blob = await response.blob();
    const fullImg = await createImageBitmap(blob);
    const tLoad = performance.now();

    // Compute letterbox scale and padding from original dimensions
    const scale = Math.min(YOLO_SIZE / fullImg.width, YOLO_SIZE / fullImg.height);
    const newW = Math.round(fullImg.width * scale);
    const newH = Math.round(fullImg.height * scale);
    const padX = (YOLO_SIZE - newW) / 2;
    const padY = (YOLO_SIZE - newH) / 2;

    // Let the browser resize (hardware-accelerated) instead of canvas drawImage
    const img = await createImageBitmap(fullImg, {
      resizeWidth: newW,
      resizeHeight: newH,
      resizeQuality: "low",
    });
    fullImg.close();

    // Letterbox to YOLO_SIZE using pre-allocated canvas
    const tensor = detectorPreprocess(img, padX, padY, newW, newH);
    img.close();
    const tPrep = performance.now();

    const detResults = await detector.run({ images: tensor });
    // Don't dispose — tensor wraps the reusable detFloat32 buffer
    const tRun = performance.now();

    // YOLOv8/v11 output: [1, 4+numClasses, numDetections]
    const output0 = detResults["output0"];
    const data = output0.data as Float32Array;
    const [, , numDetections] = output0.dims;
    output0.dispose();

    // Original image dimensions (before resize)
    const origW = newW / scale;
    const origH = newH / scale;

    let bestBbox: [number, number, number, number] | null = null;
    let bestScore = 0;

    for (let i = 0; i < numDetections; i++) {
      const birdScore = data[(4 + YOLO_BIRD_CLASS) * numDetections + i];

      if (birdScore >= msg.detectThreshold && birdScore > bestScore) {
        const cx = data[0 * numDetections + i];
        const cy = data[1 * numDetections + i];
        const w = data[2 * numDetections + i];
        const h = data[3 * numDetections + i];

        const x1 = Math.max(0, (cx - w / 2 - padX) / scale);
        const y1 = Math.max(0, (cy - h / 2 - padY) / scale);
        const x2 = Math.min(origW, (cx + w / 2 - padX) / scale);
        const y2 = Math.min(origH, (cy + h / 2 - padY) / scale);

        bestBbox = [x1, y1, x2, y2];
        bestScore = birdScore;
      }
    }

    const tEnd = performance.now();
    const filename = msg.path.split("/").pop() ?? msg.path.split("\\").pop();
    console.log(
      `[detector] ${filename} — ${bestBbox ? `bird (${(bestScore * 100).toFixed(1)}%)` : "SKIP"} | load=${(tLoad - t0).toFixed(0)}ms prep=${(tPrep - tLoad).toFixed(0)}ms run=${(tRun - tPrep).toFixed(0)}ms total=${(tEnd - t0).toFixed(0)}ms`,
    );

    post({
      type: "detect-result",
      path: msg.path,
      bbox: bestBbox,
    });
  } catch (err) {
    post({ type: "detect-error", path: msg.path, error: String(err) });
  }
}

// ---------------------------------------------------------------------------
// Dispose
// ---------------------------------------------------------------------------

function handleDispose() {
  detector?.release();
  detector = null;
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/** Letterbox pre-resized image into YOLO_SIZE canvas, reusing pre-allocated buffers. */
function detectorPreprocess(
  img: ImageBitmap,
  padX: number,
  padY: number,
  newW: number,
  newH: number,
): ort.Tensor {
  // Draw on pre-allocated canvas with gray padding
  detCtx.fillStyle = "rgb(114,114,114)"; // YOLO standard padding color
  detCtx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
  detCtx.drawImage(img, padX, padY, newW, newH);

  const imageData = detCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);
  const rgba = imageData.data;
  for (let i = 0; i < DET_PIXELS; i++) {
    const base = i * 4;
    detFloat32[i] = rgba[base] / 255;
    detFloat32[DET_PIXELS + i] = rgba[base + 1] / 255;
    detFloat32[2 * DET_PIXELS + i] = rgba[base + 2] / 255;
  }

  return new ort.Tensor("float32", detFloat32, [1, 3, YOLO_SIZE, YOLO_SIZE]);
}
