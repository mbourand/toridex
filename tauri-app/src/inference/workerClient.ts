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

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let worker: Worker | null = null;
let modelsReady = false;

let initResolve: (() => void) | null = null;
let initReject: ((e: Error) => void) | null = null;
let statusCallback: ((msg: string) => void) | null = null;

let pendingResolve: ((r: ProcessResult) => void) | null = null;
let pendingReject: ((e: Error) => void) | null = null;

// ---------------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------------

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = handleMessage;
    worker.onerror = (e) => {
      const err = new Error(e.message || "Worker error");
      initReject?.(err);
      pendingReject?.(err);
      initResolve = initReject = null;
      pendingResolve = pendingReject = null;
    };
  }
  return worker;
}

function handleMessage(e: MessageEvent) {
  const msg = e.data;
  switch (msg.type) {
    case "init-status":
      statusCallback?.(msg.status);
      break;
    case "init-done":
      modelsReady = true;
      statusCallback?.("");
      statusCallback = null;
      initResolve?.();
      initResolve = initReject = null;
      break;
    case "init-error":
      statusCallback = null;
      initReject?.(new Error(msg.error));
      initResolve = initReject = null;
      break;
    case "result":
      pendingResolve?.(msg);
      pendingResolve = pendingReject = null;
      break;
    case "error":
      pendingReject?.(new Error(msg.error));
      pendingResolve = pendingReject = null;
      break;
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Load ONNX models in the worker (cached — only loads once). */
export async function initInferenceWorker(
  detectorUrl: string,
  classifierUrl: string,
  labelMapUrl: string,
  onStatus?: (msg: string) => void,
): Promise<void> {
  if (modelsReady) return;

  return new Promise<void>((resolve, reject) => {
    initResolve = resolve;
    initReject = reject;
    statusCallback = onStatus ?? null;
    getWorker().postMessage({
      type: "init",
      detectorUrl,
      classifierUrl,
      labelMapUrl,
    });
  });
}

/** Process a single image (detect + classify) in the worker. */
export async function processImage(file: {
  url: string;
  path: string;
  folder: string;
  fileMtime: number;
  fileSize: number;
  detectThreshold: number;
  minConfidence: number;
  topK: number;
}): Promise<ProcessResult> {
  if (!modelsReady) throw new Error("Models not initialized");

  return new Promise<ProcessResult>((resolve, reject) => {
    pendingResolve = resolve;
    pendingReject = reject;
    getWorker().postMessage({ type: "process", ...file });
  });
}

/** Terminate the worker and release models. */
export function terminateInferenceWorker(): void {
  if (worker) {
    worker.postMessage({ type: "dispose" });
    worker.terminate();
    worker = null;
    modelsReady = false;
  }
}
