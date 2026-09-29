import type { ApiError } from './contracts';

export class ServiceError extends Error {
  code: string;
  requestId: string;
  constructor(message: string, code = 'NETWORK_ERROR', requestId = '') {
    super(message);
    this.name = 'ServiceError';
    this.code = code;
    this.requestId = requestId;
  }
}

export async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { ...options, cache: 'no-store' });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ServiceError('Сервер недоступен. Проверьте, что он запущен, и повторите запрос.');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const api = body as ApiError | null;
    throw new ServiceError(api?.error?.message || `Ошибка сервера (${response.status}).`, api?.error?.code || `HTTP_${response.status}`, api?.error?.request_id || '');
  }
  if (body === null) throw new ServiceError('Сервер вернул ответ в неизвестном формате.', 'INVALID_RESPONSE');
  return body as T;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос.';
}

export function csvCell(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

// Browser EXIF rotation must not silently change the raw-pixel BBox contract.
// Inspect every metadata block; malformed orientation metadata fails closed.
export async function imageOrientation(file: File): Promise<number> {
  const view = new DataView(await file.arrayBuffer());
  function tiffOrientation(start: number, end: number): number {
    if (start + 8 > end) throw new Error('Повреждены метаданные EXIF.');
    const order = view.getUint16(start);
    if (order !== 0x4949 && order !== 0x4d4d) throw new Error('Неизвестный формат метаданных EXIF.');
    const little = order === 0x4949;
    if (view.getUint16(start + 2, little) !== 42) throw new Error('Повреждены метаданные EXIF.');
    const relative = view.getUint32(start + 4, little);
    const directory = start + relative;
    if (relative < 8 || directory + 2 > end) throw new Error('Повреждены метаданные EXIF.');
    const count = view.getUint16(directory, little);
    if (directory + 2 + count * 12 > end) throw new Error('Повреждены метаданные EXIF.');
    for (let i = 0; i < count; i++) {
      const entry = directory + 2 + i * 12;
      if (view.getUint16(entry, little) !== 0x0112) continue;
      if (view.getUint16(entry + 2, little) !== 3 || view.getUint32(entry + 4, little) !== 1) throw new Error('Неизвестный формат ориентации EXIF.');
      const orientation = view.getUint16(entry + 8, little);
      if (orientation < 1 || orientation > 8) throw new Error('Некорректная ориентация EXIF.');
      if (orientation !== 1) return orientation;
    }
    return 1;
  }
  if (view.byteLength >= 8 && view.getUint32(0) === 0x89504e47 && view.getUint32(4) === 0x0d0a1a0a) {
    let offset = 8;
    while (offset + 12 <= view.byteLength) {
      const length = view.getUint32(offset);
      const type = view.getUint32(offset + 4);
      const start = offset + 8;
      if (start + length + 4 > view.byteLength) throw new Error('Повреждён файл PNG.');
      if (type === 0x65584966) {
        const orientation = tiffOrientation(start, start + length);
        if (orientation !== 1) return orientation;
      }
      offset = start + length + 4;
      if (type === 0x49454e44) return 1;
    }
    throw new Error('Повреждён файл PNG.');
  }
  if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) throw new Error('Выберите исходное изображение JPEG или PNG.');
  let offset = 2;
  while (offset < view.byteLength) {
    if (view.getUint8(offset++) !== 0xff) throw new Error('Повреждён файл JPEG.');
    while (offset < view.byteLength && view.getUint8(offset) === 0xff) offset++;
    if (offset >= view.byteLength) throw new Error('Повреждён файл JPEG.');
    const marker = view.getUint8(offset++);
    if (marker === 0xda || marker === 0xd9) return 1;
    if (marker === 0x01 || marker >= 0xd0 && marker <= 0xd7) continue;
    if (offset + 2 > view.byteLength) throw new Error('Повреждён файл JPEG.');
    const length = view.getUint16(offset);
    if (length < 2 || offset + length > view.byteLength) throw new Error('Повреждён файл JPEG.');
    if (marker === 0xe1 && length >= 8 && view.getUint32(offset + 2) === 0x45786966 && view.getUint16(offset + 6) === 0) {
      const orientation = tiffOrientation(offset + 8, offset + length);
      if (orientation !== 1) return orientation;
    }
    offset += length;
  }
  throw new Error('Повреждён файл JPEG.');
}
