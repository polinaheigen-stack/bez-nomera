import assert from 'node:assert/strict';
import { test, mock } from 'node:test';
import { csvCell, imageOrientation, request, ServiceError } from './api.ts';

function tiff(orientation, little = true) {
  const bytes = new Uint8Array(26);
  const v = new DataView(bytes.buffer);
  v.setUint16(0, little ? 0x4949 : 0x4d4d);
  v.setUint16(2, 42, little); v.setUint32(4, 8, little);
  v.setUint16(8, 1, little); v.setUint16(10, 0x0112, little);
  v.setUint16(12, 3, little); v.setUint32(14, 1, little);
  v.setUint16(18, orientation, little);
  return bytes;
}
function concat(...parts) {
  const bytes = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let offset = 0; for (const part of parts) { bytes.set(part, offset); offset += part.length; }
  return bytes;
}
function jpegSegment(marker, content) {
  const header = new Uint8Array(4); const view = new DataView(header.buffer);
  view.setUint16(0, marker); view.setUint16(2, content.length + 2);
  return concat(header, content);
}
function jpegExif(orientation, little = true) { return jpegSegment(0xffe1, concat(new Uint8Array([69, 120, 105, 102, 0, 0]), tiff(orientation, little))); }
function jpeg(...segments) { return new File([concat(new Uint8Array([255, 216]), ...segments, new Uint8Array([255, 217]))], 'fixture.jpg', { type: 'image/jpeg' }); }
function pngChunk(type, content) {
  const header = new Uint8Array(8); const view = new DataView(header.buffer);
  view.setUint32(0, content.length); view.setUint32(4, type);
  // The parser reads metadata only; browser/Pillow are responsible for CRC and pixels.
  return concat(header, content, new Uint8Array(4));
}
function png(...chunks) { return new File([concat(new Uint8Array([137,80,78,71,13,10,26,10]), ...chunks, pngChunk(0x49454e44, new Uint8Array()))], 'fixture.png', { type: 'image/png' }); }

test('CSV preserves image IDs with punctuation and newlines', () => {
  assert.equal(csvCell('vehicle-id'), 'vehicle-id');
  assert.equal(csvCell('car,"2"'), '"car,""2"""');
  assert.equal(csvCell('car\n2'), '"car\n2"');
});
for (const little of [true, false]) {
  for (let orientation = 1; orientation <= 8; orientation++) {
    test(`JPEG orientation ${orientation}, byte order ${little ? 'II' : 'MM'}`, async () => assert.equal(await imageOrientation(jpeg(jpegExif(orientation, little))), orientation));
    test(`PNG eXIf orientation ${orientation}, byte order ${little ? 'II' : 'MM'}`, async () => assert.equal(await imageOrientation(png(pngChunk(0x65584966, tiff(orientation, little)))), orientation));
  }
}
test('JPEG metadata beyond 64 KiB is still inspected', async () => {
  assert.equal(await imageOrientation(jpeg(jpegSegment(0xffe2, new Uint8Array(65530)), jpegSegment(0xffe2, new Uint8Array(65530)), jpegExif(6))), 6);
});
test('Duplicate EXIF cannot hide rotated metadata behind orientation 1', async () => {
  assert.equal(await imageOrientation(jpeg(jpegExif(1), jpegExif(8))), 8);
  assert.equal(await imageOrientation(png(pngChunk(0x65584966, tiff(1)), pngChunk(0x65584966, tiff(8)))), 8);
});
test('Invalid orientation and malformed EXIF fail closed', async () => {
  await assert.rejects(imageOrientation(jpeg(jpegExif(0))), /ориентация/);
  await assert.rejects(imageOrientation(jpeg(jpegExif(9))), /ориентация/);
  await assert.rejects(imageOrientation(png(pngChunk(0x65584966, new Uint8Array(2)))), /EXIF/);
  await assert.rejects(imageOrientation(jpeg(jpegSegment(0xffe1, new Uint8Array([69,120,105,102,0,0])))), /EXIF/);
});
test('Non-images and broken metadata containers are rejected', async () => {
  await assert.rejects(imageOrientation(new File(['not an image'], 'fake.jpg')), /JPEG или PNG/);
  await assert.rejects(imageOrientation(new File([new Uint8Array([255,216,255,225,0,100])], 'broken.jpg')), /Повреждён/);
});
test('Files without orientation metadata keep raw dimensions', async () => {
  assert.equal(await imageOrientation(jpeg()), 1);
  assert.equal(await imageOrientation(png()), 1);
});

test('Unavailable backend is an error and cannot become an empty successful response', async () => {
  const fetch = mock.method(globalThis, 'fetch', async () => { throw new TypeError('Connection refused'); });
  try { await assert.rejects(request('/api/v1/status'), (error) => error instanceof ServiceError && error.code === 'NETWORK_ERROR'); }
  finally { fetch.mock.restore(); }
});

test('Missing production model preserves the server error instead of inventing results', async () => {
  const fetch = mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ error: { code: 'MODEL_UNAVAILABLE', message: 'Model unavailable', request_id: 'test-request' } }), { status: 503 }));
  try { await assert.rejects(request('/api/v1/runs'), (error) => error instanceof ServiceError && error.code === 'MODEL_UNAVAILABLE' && error.requestId === 'test-request'); }
  finally { fetch.mock.restore(); }
});

test('Invalid successful response is an error, not zero-valued metrics', async () => {
  const fetch = mock.method(globalThis, 'fetch', async () => new Response('not JSON', { status: 200 }));
  try { await assert.rejects(request('/api/v1/report'), (error) => error instanceof ServiceError && error.code === 'INVALID_RESPONSE'); }
  finally { fetch.mock.restore(); }
});

test('device change accepts asynchronous 202 response and sends only explicit selection', async () => {
  const result = { selected_device: 'cpu', active_device: null, switching: true, can_switch: false,
    devices: [{ id: 'cuda', name: 'NVIDIA GPU', available: false, reason: 'CUDA недоступна.' }, { id: 'cpu', name: 'CPU', available: true, reason: null }], error: null };
  const fetch = mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/v1/settings/device');
    assert.equal(options.method, 'PUT');
    assert.deepEqual(JSON.parse(options.body), { device: 'cpu' });
    return new Response(JSON.stringify(result), { status: 202 });
  });
  try { assert.deepEqual(await request('/api/v1/settings/device', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: 'cpu' }) }), result); }
  finally { fetch.mock.restore(); }
});

test('device change preserves unavailable and busy errors for the settings dialog', async () => {
  for (const [status, code] of [[409, 'DEVICE_BUSY'], [409, 'DEVICE_UNAVAILABLE']]) {
    const fetch = mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ error: { code, message: 'Смена устройства недоступна.', request_id: 'device-request' } }), { status }));
    try { await assert.rejects(request('/api/v1/settings/device', { method: 'PUT' }), (error) => error instanceof ServiceError && error.code === code && error.message === 'Смена устройства недоступна.' && error.requestId === 'device-request'); }
    finally { fetch.mock.restore(); }
  }
});
