export type Mode = 'prod';
export type BBox = [number, number, number, number];
export type ComputeDevice = 'cpu' | 'cuda';
export interface DeviceSettings { selected_device: ComputeDevice; active_device: ComputeDevice | null; switching: boolean; can_switch: boolean; devices: { id: ComputeDevice; name: string; available: boolean; reason: string | null }[]; error: string | null; }
export interface ModelInfo { name: string; version: string; sha256: string | null; dimension: number | null; device: string; inference_fingerprint?: string | null; }
export interface ServiceStatus { contract_version: '1'; mode: Mode; available: boolean; reason: string | null; model: ModelInfo | null; threshold: number | null; limits: { max_images: number; max_file_mb: number; max_upload_mb: number; max_pending_runs: number }; }
export interface Gallery { id: string; name: string; count: number; processed: number; status: 'indexing' | 'ready' | 'failed'; created_at: string; error: string | null; }
export interface ImageRef { image_id: string; image_url: string; crop_url: string; bbox: BBox; width: number; height: number; }
export interface Candidate { rank: number; image: ImageRef; score: number; accepted: boolean; }
export interface QueryResult { query: ImageRef; status: 'matched' | 'rejected' | 'error'; candidates: Candidate[]; duration_ms: number; error: string | null; }
export interface RunTiming { queue_ms: number; processing_ms: number; successful_images: number; images_per_second: number | null; mean_image_ms: number | null; }
export interface Run { timing?: RunTiming | null; evaluation?: RunEvaluation; id: string; mode: Mode; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'; gallery_id: string; model: ModelInfo | null; threshold: number; total: number; processed: number; failed: number; created_at: string; results: QueryResult[]; error: string | null; export_ready: boolean; artifacts_url: string | null; evidence_url: string; }
export interface EvaluationMetrics { map_at_10: number | null; rank_1: number | null; rank_5: number | null; micro_f1: number | null; tnr: number | null; latency_ms: number | null; throughput_fps: number | null; peak_ram_mb: number | null; peak_vram_mb: number | null; }
export interface EvaluationReport { mode: Mode; measured: boolean; model: ModelInfo | null; calibration_sha256?: string | null; ground_truth_sha256?: string | null; dataset: string | null; hardware: string | null; run_id: string | null; metrics: EvaluationMetrics | null; notes: string[]; }
export interface RunEvaluation { status: 'unavailable' | 'waiting' | 'running' | 'completed' | 'failed'; report: EvaluationReport | null; error: string | null; ground_truth_sha256: string | null; }
export interface ApiError { error: { code: string; message: string; request_id: string }; }
