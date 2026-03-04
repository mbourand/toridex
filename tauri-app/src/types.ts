export interface Species {
  idx: number;
  epithet: string;
  scientificName: string;
  frenchName: string;
  occurrenceCount: number;
  referencePhotoId: number | null;
}

export interface PhotoResult {
  species_idx: number;
  scientificName: string;
  confidence: number;
  exif_date?: string;
  exif_lat?: number;
  exif_lon?: number;
  top_k?: Array<{ scientificName: string; confidence: number }>;
}

export interface UserPhoto {
  path: string;
  result: PhotoResult;
}

export interface ScanResults {
  folder: string;
  scanned_at: string;
  photos: Record<string, PhotoResult>;
}

export type FilterMode = "all" | "found" | "not-found";
export type SortMode = "name" | "rarity" | "date";
