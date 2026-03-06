export interface Species {
  idx: number;
  epithet: string;
  scientificName: string;
  frenchName: string;
  occurrenceCount: number;
  referencePhotoId: number | null;
  referencePhotoUrl: string | null;
}

export interface PhotoResult {
  scientificName: string;
  confidence: number;
  exif_date?: string;
  exif_lat?: number;
  exif_lon?: number;
  top_k?: Array<{ scientificName: string; confidence: number }>;
  thumbPath?: string;
  userSpecies?: string;
  modelSpecies?: string;
}

export interface UserPhoto {
  path: string;
  result: PhotoResult;
}

export interface AppConfig {
  folders: string[];
}

export interface FileToProcess {
  path: string;
  folder: string;
  fileMtime: number;
  fileSize: number;
}

export interface PreparedScan {
  toProcess: FileToProcess[];
  skippedCount: number;
}

export interface ModelPaths {
  detector: string;
  classifier: string;
  labelMap: string;
}

export interface FullRescanInfo {
  purgedCount: number;
  totalRemaining: number;
}

export interface LabelConflict {
  path: string;
  modelSpecies: string;
  modelConfidence: number;
  userSpecies: string;
  thumbPath: string | null;
}

export type FilterMode = "all" | "found" | "not-found";
export type SortMode = "name" | "rarity" | "date";
