export interface ProcessResult {
  path: string;
  folder: string;
  species: string;
  speciesIdx: number;
  confidence: number;
  fileMtime: number;
  fileSize: number;
  topKJson?: string;
}

export interface DetectResult {
  path: string;
  bbox: [number, number, number, number] | null;
}

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let detectorWorker: Worker | null = null;
let classifierWorker: Worker | null = null;
let modelsReady = false;

// Init callbacks (one per worker)
let detInitResolve: (() => void) | null = null;
let detInitReject: ((e: Error) => void) | null = null;
let clsInitResolve: (() => void) | null = null;
let clsInitReject: ((e: Error) => void) | null = null;
let statusCallback: ((msg: string) => void) | null = null;

// Pending detect/classify callbacks
let detPendingResolve: ((r: DetectResult) => void) | null = null;
let detPendingReject: ((e: Error) => void) | null = null;
let clsPendingResolve: ((r: ProcessResult) => void) | null = null;
let clsPendingReject: ((e: Error) => void) | null = null;

// ---------------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------------

function getDetectorWorker(): Worker {
  if (!detectorWorker) {
    detectorWorker = new Worker(
      new URL("./detectorWorker.ts", import.meta.url),
      { type: "module" },
    );
    detectorWorker.onmessage = handleDetectorMessage;
    detectorWorker.onerror = (e) => {
      const err = new Error(e.message || "Detector worker error");
      detInitReject?.(err);
      detPendingReject?.(err);
      detInitResolve = detInitReject = null;
      detPendingResolve = detPendingReject = null;
    };
  }
  return detectorWorker;
}

function getClassifierWorker(): Worker {
  if (!classifierWorker) {
    classifierWorker = new Worker(
      new URL("./classifierWorker.ts", import.meta.url),
      { type: "module" },
    );
    classifierWorker.onmessage = handleClassifierMessage;
    classifierWorker.onerror = (e) => {
      const err = new Error(e.message || "Classifier worker error");
      clsInitReject?.(err);
      clsPendingReject?.(err);
      clsInitResolve = clsInitReject = null;
      clsPendingResolve = clsPendingReject = null;
    };
  }
  return classifierWorker;
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------

function handleDetectorMessage(e: MessageEvent) {
  const msg = e.data;
  switch (msg.type) {
    case "init-status":
      statusCallback?.(msg.status);
      break;
    case "init-done":
      detInitResolve?.();
      detInitResolve = detInitReject = null;
      break;
    case "init-error":
      detInitReject?.(new Error(msg.error));
      detInitResolve = detInitReject = null;
      break;
    case "detect-result":
      detPendingResolve?.(msg);
      detPendingResolve = detPendingReject = null;
      break;
    case "detect-error":
      detPendingReject?.(new Error(msg.error));
      detPendingResolve = detPendingReject = null;
      break;
  }
}

function handleClassifierMessage(e: MessageEvent) {
  const msg = e.data;
  switch (msg.type) {
    case "init-status":
      statusCallback?.(msg.status);
      break;
    case "init-done":
      clsInitResolve?.();
      clsInitResolve = clsInitReject = null;
      break;
    case "init-error":
      clsInitReject?.(new Error(msg.error));
      clsInitResolve = clsInitReject = null;
      break;
    case "classify-result":
      clsPendingResolve?.(msg);
      clsPendingResolve = clsPendingReject = null;
      break;
    case "classify-error":
      clsPendingReject?.(new Error(msg.error));
      clsPendingResolve = clsPendingReject = null;
      break;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Load models in both workers (cached — only loads once). */
export async function initInferenceWorker(
  detectorUrl: string,
  classifierUrl: string,
  labelMapUrl: string,
  onStatus?: (msg: string) => void,
): Promise<void> {
  if (modelsReady) return;

  statusCallback = onStatus ?? null;

  // Init both workers in parallel
  const detReady = new Promise<void>((resolve, reject) => {
    detInitResolve = resolve;
    detInitReject = reject;
    getDetectorWorker().postMessage({ type: "init", detectorUrl });
  });

  const clsReady = new Promise<void>((resolve, reject) => {
    clsInitResolve = resolve;
    clsInitReject = reject;
    getClassifierWorker().postMessage({
      type: "init",
      classifierUrl,
      labelMapUrl,
    });
  });

  await Promise.all([detReady, clsReady]);
  modelsReady = true;
  statusCallback?.("");
  statusCallback = null;
}

/** Run bird detection on an image. Returns bbox or null. */
export async function detectBird(input: {
  url: string;
  path: string;
  detectThreshold: number;
}): Promise<DetectResult> {
  if (!modelsReady) throw new Error("Models not initialized");

  return new Promise<DetectResult>((resolve, reject) => {
    detPendingResolve = resolve;
    detPendingReject = reject;
    getDetectorWorker().postMessage({ type: "detect", ...input });
  });
}

/** Run species classification on a cropped bird image. */
export async function classifyBird(input: {
  url: string;
  path: string;
  folder: string;
  fileMtime: number;
  fileSize: number;
  bbox: [number, number, number, number];
  minConfidence: number;
  topK: number;
}): Promise<ProcessResult> {
  if (!modelsReady) throw new Error("Models not initialized");

  return new Promise<ProcessResult>((resolve, reject) => {
    clsPendingResolve = resolve;
    clsPendingReject = reject;
    getClassifierWorker().postMessage({ type: "classify", ...input });
  });
}

/** Terminate both workers and release models. */
export function terminateInferenceWorker(): void {
  if (detectorWorker) {
    detectorWorker.postMessage({ type: "dispose" });
    detectorWorker.terminate();
    detectorWorker = null;
  }
  if (classifierWorker) {
    classifierWorker.postMessage({ type: "dispose" });
    classifierWorker.terminate();
    classifierWorker = null;
  }
  modelsReady = false;
}
