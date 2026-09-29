import type { BBox } from './contracts';

export type SingleCoordinates = { imageId: string; bbox: BBox };

// File selection resets previousSize to [0, 0]. Remounting the editor after a
// tab change does not select a new image and must retain CSV or manual edits.
export function bboxAfterImageLoad(bbox: BBox, previousSize: readonly [number, number], loadedSize: readonly [number, number]): BBox {
  const [width, height] = loadedSize;
  if (width > 0 && height > 0 && previousSize[0] === width && previousSize[1] === height) return bbox;
  return [0, 0, width, height];
}

// CSV quoting, escaped quotes and UTF-8 BOM follow the same input format as the server.
export function parseSingleCoordinates(source: string): SingleCoordinates {
  const text = source.replace(/^\uFEFF/, '');
  const rows: string[][] = [];
  let row: string[] = [], cell = '', quoted = false, closed = false;
  for (let i = 0; i <= text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === undefined) throw new Error('CSV содержит незакрытую кавычку.');
      if (char === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else { quoted = false; closed = true; }
      } else cell += char;
    } else if (char === ',' || char === '\r' || char === '\n' || char === undefined) {
      const blank = row.length === 0 && cell === '' && !closed;
      row.push(cell); cell = ''; closed = false;
      if (char !== ',') {
        if (!blank) rows.push(row);
        row = [];
        if (char === '\r' && text[i + 1] === '\n') i++;
      }
    } else if (char === '"' && cell === '' && !closed) quoted = true;
    else {
      if (closed || char === '"') throw new Error('Некорректные кавычки в CSV.');
      cell += char;
    }
  }
  if (rows[0]?.length !== 5 || rows[0].join(',') !== 'image_id,x,y,w,h') {
    throw new Error('Ожидается заголовок image_id,x,y,w,h. Разделитель — запятая; кодировка — UTF-8.');
  }
  if (rows.length !== 2) throw new Error('Для одного снимка CSV должен содержать ровно одну строку с автомобилем. Для нескольких снимков выберите «Пакет».');
  if (rows[1].length !== 5) throw new Error('В строке CSV должно быть пять полей: image_id,x,y,w,h.');
  const [imageId, ...values] = rows[1];
  if (!imageId || imageId.trim() !== imageId) throw new Error('Укажите image_id — имя снимка без расширения и пробелов по краям.');
  const bbox = values.map(value => {
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value.trim())) throw new Error('Координаты рамки должны быть целыми пикселями.');
    const number = Number(value);
    if (!Number.isSafeInteger(number)) throw new Error('Координаты рамки должны быть целыми пикселями.');
    return number;
  }) as BBox;
  if (bbox[0] < 0 || bbox[1] < 0 || bbox[2] <= 0 || bbox[3] <= 0) throw new Error('Рамка требует x, y не меньше нуля и положительные ширину и высоту.');
  return { imageId, bbox };
}

export function coordinatesIssue(row: SingleCoordinates, filename: string, width: number, height: number): string {
  const imageId = filename.replace(/\.[^.]+$/, '');
  if (row.imageId !== imageId) return `В CSV указан «${row.imageId}», а выбран снимок «${imageId}». Загрузите CSV для этого снимка.`;
  if (row.bbox[0] + row.bbox[2] > width || row.bbox[1] + row.bbox[3] > height) return `Рамка из CSV выходит за границы снимка (${width} × ${height} пикселей). Исправьте CSV или задайте рамку вручную.`;
  return '';
}
