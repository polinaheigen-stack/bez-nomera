import { useEffect, useRef, useState, type FormEvent, type PointerEvent, type ReactNode } from 'react';
import { ArrowDownToLine, ArrowRight, Check, ChevronDown, ChevronRight, CircleHelp, Clock3, Expand, FileImage, FileSpreadsheet, FolderPlus, ImagePlus, Layers3, LoaderCircle, Monitor, Moon, PanelTop, Plus, RefreshCw, ScanLine, Search, Settings, ShieldCheck, Square, Sun, X } from 'lucide-react';
import { csvCell, errorMessage, imageOrientation, request, ServiceError } from './api';
import type { BBox, Gallery, ImageRef, Run, ServiceStatus } from './contracts';

import StatusNotice from './StatusNotice';
import Report from './Verification';
import { readRunSelection, saveRunSelection } from './runSelection';
import CsvHelp from './CsvHelp';
import RunStats from './RunStats';
import DeviceSettingsForm from './DeviceSettingsForm';
import useDeviceSettings from './useDeviceSettings';
import { bboxAfterImageLoad, coordinatesIssue, parseSingleCoordinates, type SingleCoordinates } from './singleCoordinates';

type Theme = 'light' | 'dark' | 'system';
type Tab = 'search' | 'report';
type Zoom = { image: ImageRef; title: string };
const ACTIVE = new Set(['queued', 'running']);
const baseUrl = '/api/v1';
const runLabels: Record<Run['status'], string> = { queued: 'В очереди', running: 'Обработка', completed: 'Завершён', failed: 'Ошибка', cancelled: 'Отменён' };

export function reconcileRunSelection(current: Run | null, history: Run[], requestedGeneration: number, currentGeneration: number, confirmedMissingId: string | null = null, allowAuto = true): Run | null {
  if (requestedGeneration !== currentGeneration) return current;
  if (current && current.id !== confirmedMissingId) return current;
  return allowAuto ? history.find((item) => item.status === 'running') || history.find((item) => item.status === 'queued') || null : null;
}

function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    try { const saved = localStorage.getItem('vehicle-theme'); return saved === 'light' || saved === 'dark' ? saved : 'system'; } catch { return 'system'; }
  });
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => { const resolved = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme; document.documentElement.dataset.theme = resolved; document.documentElement.style.colorScheme = resolved; };
    apply(); media.addEventListener('change', apply);
    try { localStorage.setItem('vehicle-theme', theme); } catch { /* Private browsing can restrict storage. */ }
    return () => media.removeEventListener('change', apply);
  }, [theme]);
  return [theme, setTheme] as const;
}

function useObjectURL(file: File | null) {
  const [url, setUrl] = useState('');
  useEffect(() => { if (!file) { setUrl(''); return; } const next = URL.createObjectURL(file); setUrl(next); return () => URL.revokeObjectURL(next); }, [file]);
  return url;
}

function score(value: number) { return new Intl.NumberFormat('ru', { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(value); }
function time(value: number) { return new Intl.NumberFormat('ru', { maximumFractionDigits: 1 }).format(value); }
function date(value: string) { return new Date(value).toLocaleTimeString('ru', { hour: '2-digit', minute: '2-digit' }); }

function Modal({ title, children, onClose, wide = false }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { const dialog = ref.current; dialog?.showModal(); return () => dialog?.close(); }, []);
  return <dialog ref={ref} className={`modal ${wide ? 'modal-wide' : ''}`} onCancel={(event) => { event.preventDefault(); onClose(); }} aria-label={title}>
    <div className="modal-content"><header className="modal-heading"><h2>{title}</h2><button className="icon-button" type="button" aria-label="Закрыть окно" onClick={onClose}><X size={20} /></button></header>{children}</div>
  </dialog>;
}

function Alert({ children, kind = 'error', onClose }: { children: ReactNode; kind?: 'error' | 'info' | 'warning'; onClose?: () => void }) {
  return <div className={`alert alert-${kind}`} role={kind === 'error' ? 'alert' : 'status'}><CircleHelp size={18} /><div>{children}</div>{onClose && <button className="icon-button" type="button" aria-label="Скрыть сообщение" onClick={onClose}><X size={16} /></button>}</div>;
}

function UploadField({ label, accept, multiple, onChange, files }: { label: string; accept: string; multiple?: boolean; onChange: (files: File[]) => void; files: File[] }) {
  return <label className={`upload-field ${files.length ? 'upload-filled' : ''}`}>
    {accept.includes('csv') ? <FileSpreadsheet size={22} /> : <ImagePlus size={24} />}
    <strong>{files.length ? (files.length === 1 ? files[0].name : `${files.length} изображений`) : label}</strong>
    <span>{files.length ? 'Нажмите, чтобы заменить' : accept.includes('csv') ? 'image_id, x, y, w, h' : 'JPEG или PNG'}</span>
    <input aria-label={label} type="file" accept={accept} multiple={multiple} onChange={(event) => { const selected = Array.from(event.target.files || []); event.target.value = ''; if (selected.length) onChange(selected); }} />
  </label>;
}

function BBoxEditor({ file, bbox, setBBox, onSize }: { file: File; bbox: BBox; setBBox: (bbox: BBox) => void; onSize: (width: number, height: number) => void }) {
  const url = useObjectURL(file);
  const img = useRef<HTMLImageElement>(null);
  const [size, setSize] = useState<[number, number]>([1, 1]);
  const anchor = useRef<[number, number] | null>(null);
  function point(event: PointerEvent<HTMLDivElement>): [number, number] {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [Math.max(0, Math.min(size[0], Math.round((event.clientX - bounds.left) / bounds.width * size[0]))), Math.max(0, Math.min(size[1], Math.round((event.clientY - bounds.top) / bounds.height * size[1])))];
  }
  return <div className="bbox-editor">
    <div className="bbox-image" style={{ aspectRatio: `${size[0]} / ${size[1]}` }} onPointerDown={(event) => { anchor.current = point(event); event.currentTarget.setPointerCapture(event.pointerId); }} onPointerMove={(event) => { if (!anchor.current) return; const p = point(event); setBBox([Math.min(p[0], anchor.current[0]), Math.min(p[1], anchor.current[1]), Math.abs(p[0] - anchor.current[0]), Math.abs(p[1] - anchor.current[1])]); }} onPointerUp={() => { anchor.current = null; }} onPointerCancel={() => { anchor.current = null; }} title="Выделите автомобиль мышью или введите координаты ниже">
      <img ref={img} src={url} alt="Исходный снимок: выделите рамку автомобиля" onLoad={(event) => { const { naturalWidth: w, naturalHeight: h } = event.currentTarget; setSize([w, h]); onSize(w, h); }} onError={() => onSize(0, 0)} draggable={false} />
      <svg viewBox={`0 0 ${size[0]} ${size[1]}`} aria-hidden="true"><rect x={bbox[0]} y={bbox[1]} width={bbox[2]} height={bbox[3]} fill="rgba(101,198,240,.08)" stroke="#65C6F0" strokeWidth={Math.max(2, size[0] / 160)} /></svg>
    </div>
    <p className="field-hint">Выделите автомобиль на снимке или задайте рамку в пикселях исходного кадра.</p>
    <div className="bbox-fields">{(['X', 'Y', 'Ширина', 'Высота'] as const).map((label, i) => <label key={label}><span>{label}</span><input type="number" min={i > 1 ? 1 : 0} step="1" aria-label={`Рамка: ${label}`} value={bbox[i]} onChange={(event) => { const next = [...bbox] as BBox; next[i] = Math.max(0, Math.trunc(Number(event.target.value))); setBBox(next); }} /></label>)}</div>
    <button className="text-button" type="button" onClick={() => setBBox([0, 0, size[0], size[1]])}>Весь кадр</button>
  </div>;
}

function Photo({ image, title, subtitle, cropped, onZoom }: { image: ImageRef; title: string; subtitle: string; cropped: boolean; onZoom: (image: ImageRef, title: string) => void }) {
  return <section className="photo-card"><div className="photo-heading"><span>{title}</span><span className="muted">{subtitle}</span></div>
    <button className="photo-button" onClick={() => onZoom(image, title)} type="button" aria-label={`Увеличить: ${title}`}><img src={cropped ? image.crop_url : image.image_url} alt={`${title}, ${image.image_id}`} /><span className="photo-expand"><Expand size={17} /> Увеличить</span></button>
    <div className="photo-footer"><span title={image.image_id}>{image.image_id}</span><span>{cropped ? `${image.bbox[2]} × ${image.bbox[3]}` : `${image.width} × ${image.height}`} px</span></div>
  </section>;
}

function Results({ run, queryIndex, setQueryIndex, selectedRank, setSelectedRank, cropped, onZoom }: { run: Run | null; queryIndex: number; setQueryIndex: (index: number) => void; selectedRank: number; setSelectedRank: (rank: number) => void; cropped: boolean; onZoom: (image: ImageRef, title: string) => void }) {
  const result = run?.results[queryIndex];
  const selected = result?.candidates.find((candidate) => candidate.rank === selectedRank) || result?.candidates[0];
  const [expanded, setExpanded] = useState(false);
  useEffect(() => { setExpanded(false); }, [queryIndex]);
  return <>
    {run && <div className="run-status" aria-live="polite"><span className={`status-dot ${ACTIVE.has(run.status) ? 'pulse' : run.status === 'completed' ? 'success' : ''}`} /><strong>{runLabels[run.status]}</strong><span>{run.processed} из {run.total} снимков</span>{run.failed > 0 && <span className="danger-text">Ошибок: {run.failed}</span>}<progress aria-label="Обработано снимков" value={run.processed} max={Math.max(1, run.total)} /></div>}
    {run && <RunStats run={run} />}
    {run && run.total > 1 && <div className="query-strip" role="group" aria-label="Снимки пакета">{run.results.map((item, index) => <button key={item.query.image_id} className={queryIndex === index ? 'selected' : ''} onClick={() => { setQueryIndex(index); setSelectedRank(1); }} type="button" aria-label={`Запрос ${index + 1}: ${item.query.image_id}`} aria-pressed={queryIndex === index}><img src={item.query.crop_url} alt="" loading="lazy" decoding="async" /><span>{index + 1}</span>{item.status === 'error' && <span className="query-error">!</span>}</button>)}{run.results.length < run.total && <span className="query-pending">Ожидают: {run.total - run.results.length}</span>}</div>}
    {result?.status === 'rejected' && <Alert kind="warning"><strong>Уверенное совпадение не найдено.</strong><br />Ниже — ближайшие изображения. Они не приняты системой как совпадения.</Alert>}
    {result?.status === 'error' && <Alert><strong>Снимок не обработан.</strong> {result.error || 'Ошибка обработки изображения.'}</Alert>}
    {run?.error && <Alert>{run.error}</Alert>}
    {!run && <p className="empty-caption"><Search size={17} /><span>Добавьте галерею, загрузите снимок и выделите автомобиль для поиска.</span></p>}
    {run && !result && ACTIVE.has(run.status) && <p className="empty-caption" role="status"><LoaderCircle size={17} className="spin" /><span>Ожидаем обработки первого снимка.</span></p>}
    {result && <p className="query-caption">Снимок {queryIndex + 1} из {run?.total}: <strong>{result.query.image_id}</strong></p>}
    {result && <div className={`comparison-grid ${selected ? '' : 'comparison-single'}`}>
      <Photo image={result.query} title="Ищем" subtitle="Запрос" cropped={cropped} onZoom={onZoom} />
      {selected && <Photo image={selected.image} title={`Кандидат №${selected.rank}`} subtitle={`${selected.rank === 1 ? "Оценка совпадения" : "Сходство"} ${score(selected.score)}`} cropped={cropped} onZoom={onZoom} />}
    </div>}
    {result && result.candidates.length > 0 ? <section className="candidates" aria-label="Результаты поиска"><div className="section-heading"><h2>{result.status === 'matched' ? 'Кандидаты по сходству' : 'Ближайшие изображения'}</h2><span>{result.candidates.length} результатов · {time(result.duration_ms)} мс</span></div>
      <div className="top-three">{result.candidates.slice(0, 3).map((candidate) => <button className={`candidate-card ${selected?.rank === candidate.rank ? 'selected' : ''}`} key={candidate.rank} aria-pressed={selected?.rank === candidate.rank} type="button" onClick={() => setSelectedRank(candidate.rank)} aria-label={`Выбрать результат ${candidate.rank}, оценка ${score(candidate.score)}`}><div className="candidate-image"><img src={cropped ? candidate.image.crop_url : candidate.image.image_url} alt={`Кандидат ${candidate.rank}`} /><span className="rank-tag">{candidate.rank.toString().padStart(2, '0')}</span>{selected?.rank === candidate.rank && <span className="selection-check"><Check size={15} /></span>}</div><div className="candidate-info"><span title={candidate.image.image_id}>{candidate.image.image_id}</span><strong>{score(candidate.score)}</strong></div><div className={`candidate-accept ${candidate.accepted ? 'accepted' : ''}`}>{candidate.accepted ? 'Принят по порогу' : candidate.rank === 1 ? 'Ниже порога' : 'Альтернативный кандидат'}</div></button>)}</div>
      {result.candidates.length > 3 && <><button className="more-results" type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? 'Свернуть дополнительные результаты' : `Показать результаты 4–${result.candidates.length}`}<ChevronDown size={17} className={expanded ? 'rotate' : ''} /></button>{expanded && <div className="remaining-results">{result.candidates.slice(3).map((candidate) => <button className={selected?.rank === candidate.rank ? 'selected' : ''} key={candidate.rank} type="button" aria-pressed={selected?.rank === candidate.rank} onClick={() => setSelectedRank(candidate.rank)}><span className="list-rank">{candidate.rank.toString().padStart(2, '0')}</span><span className="list-id">{candidate.image.image_id}</span><span>{candidate.accepted ? 'Принят' : candidate.rank === 1 ? 'Ниже порога' : 'Альтернатива'}</span><strong>{score(candidate.score)}</strong><ArrowRight size={16} /></button>)}</div>}</>}
      <p className="score-note"><CircleHelp size={14} /> У первого кандидата показана оценка совпадения с учётом похожих машин в галерее, у остальных — сходство. Это не вероятность; порядок результатов определяет сервер.</p>
    </section> : null}
    {run && <details className="technical-details"><summary><ShieldCheck size={16} /> Данные запуска и доказательства <ChevronDown size={16} /></summary><dl><div><dt>Запуск</dt><dd>{run.id}</dd></div><div><dt>Режим</dt><dd>{run.mode.toUpperCase()}</dd></div><div><dt>Модель</dt><dd>{run.model ? `${run.model.name} · ${run.model.version}` : 'Не подключена'}</dd></div><div><dt>Галерея</dt><dd>{run.gallery_id}</dd></div><div><dt>Порог принятия</dt><dd>{score(run.threshold)}</dd></div><div><dt>Вектор признаков</dt><dd>{run.model?.dimension ? `${run.model.dimension} значений` : 'Не указан'}</dd></div></dl><a className="text-link" href={run.evidence_url} target="_blank" rel="noreferrer"><ArrowDownToLine size={15} /> Открыть журнал и контрольные суммы (JSON)</a></details>}
  </>;
}

function GalleryModal({ baseUrl, signal, disabled, onClose, onCreated }: { baseUrl: string; signal: AbortSignal; disabled: boolean; onClose: () => void; onCreated: (gallery: Gallery) => void }) {
  const [name, setName] = useState('');
  const [images, setImages] = useState<File[]>([]);
  const [csv, setCSV] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const uploadController = useRef<AbortController | null>(null);
  useEffect(() => {
    const controller = new AbortController(); uploadController.current = controller;
    const abort = () => controller.abort();
    if (signal.aborted) abort();
    else signal.addEventListener('abort', abort, { once: true });
    return () => { controller.abort(); signal.removeEventListener('abort', abort); };
  }, [signal]);
  function close() { uploadController.current?.abort(); onClose(); }
  async function submit(event: FormEvent) {
    event.preventDefault();
    const uploadSignal = uploadController.current?.signal;
    if (!uploadSignal || uploadSignal.aborted || busy || disabled) return;
    setBusy(true); setError('');
    const data = new FormData(); data.append('name', name.trim()); data.append('csv', csv[0]); images.forEach((file) => data.append('images', file));
    try { const gallery = await request<Gallery>(`${baseUrl}/galleries`, { method: 'POST', body: data, signal: uploadSignal }); if (!uploadSignal.aborted) onCreated(gallery); } catch (err) { if (!uploadSignal.aborted) setError(errorMessage(err)); } finally { if (!uploadSignal.aborted) setBusy(false); }
  }
  return <Modal title="Добавить галерею" onClose={close}><form onSubmit={submit} className="gallery-form"><p className="muted">Набор фотографий, среди которых система ищет автомобиль. После создания состав галереи не меняется.</p><label className="field-label">Название<input required maxLength={80} value={name} onChange={(event) => setName(event.target.value)} placeholder="Например, проверочная галерея" /></label><UploadField label="Изображения галереи" accept="image/jpeg,image/png" multiple files={images} onChange={setImages} /><UploadField label="Файл координат галереи (.csv)" accept=".csv,text/csv" files={csv} onChange={setCSV} /><p className="field-hint">Не менее 10 фотографий и координаты автомобилей на них. Имя файла без расширения совпадает с image_id.</p><CsvHelp />{error && <Alert>{error}</Alert>}{busy && <p className="field-hint">Если сервер уже принял файлы, создание галереи продолжится после закрытия окна.</p>}<button className="button button-primary" type="submit" disabled={disabled || busy || !name.trim() || images.length < 10 || !csv.length}>{busy ? <LoaderCircle size={18} className="spin" /> : <FolderPlus size={18} />}{busy ? 'Загрузка…' : 'Создать галерею'}</button></form></Modal>;
}

export default function App() {
  const [initialSelection] = useState(readRunSelection);
  const [theme, setTheme] = useTheme();
  const [tab, setTab] = useState<Tab>('search');
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const [galleries, setGalleries] = useState<Gallery[]>([]);
  const [galleryId, setGalleryId] = useState('');
  const [history, setHistory] = useState<Run[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [queryIndex, setQueryIndex] = useState(0);
  const [selectedRank, setSelectedRank] = useState(1);
  const [cropped, setCropped] = useState(true);
  const [inputMode, setInputMode] = useState<'single' | 'batch'>('single');
  const [files, setFiles] = useState<File[]>([]);
  const [csv, setCSV] = useState<File[]>([]);
  const [singleCSV, setSingleCSV] = useState<File | null>(null);
  const [csvRow, setCsvRow] = useState<SingleCoordinates | null>(null);
  const [csvReading, setCsvReading] = useState(false);
  const [csvError, setCsvError] = useState('');
  const csvGeneration = useRef(0);
  const [bbox, setBBox] = useState<BBox>([0, 0, 1, 1]);
  const [imageSize, setImageSize] = useState<[number, number]>([0, 0]);
  const [busy, setBusy] = useState(false);
  const [uploadingCount, setUploadingCount] = useState<number | null>(initialSelection.pendingCount);
  const [uploadError, setUploadError] = useState(initialSelection.pendingCount !== null ? 'Страница открыта заново до получения подтверждения отправки. Проверьте историю запусков.' : '');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [requestId, setRequestId] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [zoom, setZoom] = useState<Zoom | null>(null);
  const [galleryModal, setGalleryModal] = useState(false);
  const [settingsModal, setSettingsModal] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [noticeRequest, setNoticeRequest] = useState(0);
  const device = useDeviceSettings(baseUrl, () => setRefreshKey((value) => value + 1));
  const session = useRef(new AbortController());
  const historyGeneration = useRef(0);
  const runRef = useRef<Run | null>(null);
  const uploadAttempt = useRef(initialSelection.pendingCount !== null);
  const rememberedRunId = useRef(initialSelection.runId);
  const galleryChosen = useRef(false);
  const formTouched = useRef(false);
  const fileGeneration = useRef(0);
  const gallery = galleries.find((item) => item.id === galleryId);
  const active = !!run && ACTIVE.has(run.status);
  const deviceChanging = device.saving || !!device.settings?.switching;
  const deviceBlocked = busy || active || history.some((item) => ACTIVE.has(item.status)) || galleries.some((item) => item.status === 'indexing');
  const ready = !!status?.available && !deviceChanging && gallery?.status === 'ready';
  useEffect(() => { runRef.current = run; }, [run]);
  useEffect(() => {
    if (run?.id) { rememberedRunId.current = run.id; saveRunSelection({ runId: run.id, pendingCount: null }); }
  }, [run?.id]);

  useEffect(() => {
    const controller = new AbortController(); session.current = controller;
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function refresh() {
      const generation = historyGeneration.current;
      try {
        const [nextStatus, nextGalleries, nextHistory] = await Promise.all([
          request<ServiceStatus>(`${baseUrl}/status`, { signal: controller.signal }), request<Gallery[]>(`${baseUrl}/galleries`, { signal: controller.signal }), request<Run[]>(`${baseUrl}/runs`, { signal: controller.signal }),
        ]);
        if (controller.signal.aborted) return;
        if (nextStatus.contract_version !== '1' || nextStatus.mode !== 'prod' || typeof nextStatus.available !== 'boolean'
          || !Array.isArray(nextGalleries) || !Array.isArray(nextHistory) || nextHistory.some((item) => item.mode !== 'prod')) {
          throw new ServiceError('Ответ сервера не соответствует рабочему интерфейсу.', 'INVALID_RESPONSE');
        }
        const selected = runRef.current;
        const selectionId = selected?.id || rememberedRunId.current;
        let restored = !selected && selectionId ? nextHistory.find((item) => item.id === selectionId) || null : null;
        let missingId: string | null = null;
        if (selectionId && generation === historyGeneration.current && !nextHistory.some((item) => item.id === selectionId)) {
          try {
            const existing = await request<Run>(`${baseUrl}/runs/${encodeURIComponent(selectionId)}`, { signal: controller.signal });
            if (existing.id !== selectionId || existing.mode !== 'prod') throw new ServiceError('Не удалось подтвердить выбранный запуск.', 'INVALID_RESPONSE');
            if (!selected) restored = existing;
          } catch (err) {
            if (err instanceof ServiceError && ['RUN_NOT_FOUND', 'HTTP_404'].includes(err.code)) missingId = selectionId;
            else throw err;
          }
        }
        if (controller.signal.aborted) return;
        if (missingId && generation === historyGeneration.current) { rememberedRunId.current = null; saveRunSelection({ runId: null, pendingCount: null }); }
        setStatus(nextStatus); setGalleries(nextGalleries);
        // A server restart may remove the selected run. A history request made
        // before a new POST/selection must not clear that newly accepted run.
        setHistory((old) => generation === historyGeneration.current ? nextHistory : old);
        setRun((old) => !old && restored && generation === historyGeneration.current && !uploadAttempt.current ? restored : reconcileRunSelection(old, nextHistory, generation, historyGeneration.current, missingId, !uploadAttempt.current));
        setConnectionError('');
        const recovered = restored || (!selected || selected.id === missingId ? nextHistory.find((item) => item.status === 'running') || nextHistory.find((item) => item.status === 'queued') : null);
        if (recovered && generation === historyGeneration.current && !uploadAttempt.current && !formTouched.current) setInputMode(recovered.total > 1 ? 'batch' : 'single');
        setGalleryId((old) => !galleryChosen.current && !formTouched.current && !uploadAttempt.current && generation === historyGeneration.current && recovered && nextGalleries.some((item) => item.id === recovered.gallery_id) ? recovered.gallery_id : nextGalleries.some((item) => item.id === old) ? old : nextGalleries.find((item) => item.status === 'ready')?.id || nextGalleries[0]?.id || '');
      } catch (err) { if (!controller.signal.aborted) { setStatus(null); setConnectionError(errorMessage(err)); } } finally { if (!controller.signal.aborted) { setLoading(false); timer = window.setTimeout(() => void refresh(), 4000); } }
    }
    void refresh();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refreshKey]);

  const runId = run?.id;
  const runStatus = run?.status;
  useEffect(() => { if (!runId) setZoom(null); }, [runId]);
  useEffect(() => {
    if (!runId || !runStatus || !ACTIVE.has(runStatus)) return;
    const controller = new AbortController(); const id = runId;
    let timer: number;
    async function poll() {
      try { const next = await request<Run>(`${baseUrl}/runs/${encodeURIComponent(id)}`, { signal: controller.signal }); if (!controller.signal.aborted) { setRun((old) => old?.id === id ? next : old); setHistory((old) => old.map((item) => item.id === id ? next : item)); } }
      catch (err) { if (!controller.signal.aborted) setError(errorMessage(err)); }
      finally { if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), 700); }
    }
    timer = window.setTimeout(() => void poll(), 250);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [runId, runStatus]);

  const singleCsvIssue = csvError || (csvRow && files[0] && imageSize[0] > 0 ? coordinatesIssue(csvRow, files[0].name, ...imageSize) : '');
  useEffect(() => {
    if (csvRow && files[0] && imageSize[0] > 0 && !singleCsvIssue) setBBox([...csvRow.bbox]);
  }, [csvRow, files, imageSize[0], imageSize[1], singleCsvIssue]);

  function clearSingleCSV() {
    csvGeneration.current++; setSingleCSV(null); setCsvRow(null); setCsvReading(false); setCsvError('');
  }
  function editBBox(next: BBox) { formTouched.current = true; clearSingleCSV(); setBBox(next); }
  function changeInputMode(next: 'single' | 'batch') {
    formTouched.current = true;
    fileGeneration.current++; clearSingleCSV(); setInputMode(next); setFiles([]); setCSV([]); setImageSize([0, 0]);
  }
  async function selectSingleCSV(next: File[]) {
    formTouched.current = true;
    const file = next[0]; if (!file) return;
    const generation = ++csvGeneration.current;
    setSingleCSV(file); setCsvRow(null); setCsvError(''); setCsvReading(true);
    try {
      if (file.size > 1024 * 1024) throw new Error('CSV для одного снимка должен быть не больше 1 МБ.');
      const bytes = await file.arrayBuffer();
      if (generation !== csvGeneration.current) return;
      let text: string;
      try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
      catch { throw new Error('Сохраните CSV в кодировке UTF-8.'); }
      setCsvRow(parseSingleCoordinates(text));
    } catch (err) { if (generation === csvGeneration.current) setCsvError(errorMessage(err)); }
    finally { if (generation === csvGeneration.current) setCsvReading(false); }
  }

  function acceptRun(next: Run) { if (next.mode !== 'prod') throw new ServiceError('Получен результат неизвестного режима.', 'INVALID_RESPONSE'); historyGeneration.current++; uploadAttempt.current = false; setUploadingCount(null); setUploadError(''); setRun(next); setGalleryId(next.gallery_id); setQueryIndex(0); setSelectedRank(1); setError(''); setRequestId(''); setHistory((old) => [next, ...old.filter((item) => item.id !== next.id)]); }
  function showError(err: unknown) { setError(errorMessage(err)); setRequestId(err instanceof ServiceError ? err.requestId : ''); }
  async function startRun(event: FormEvent) {
    event.preventDefault(); if (!ready || !fileReady || busy || active) return;
    if (inputMode === 'single' && (bbox[2] < 1 || bbox[3] < 1 || bbox[0] < 0 || bbox[1] < 0 || bbox[0] + bbox[2] > imageSize[0] || bbox[1] + bbox[3] > imageSize[1])) { setError('Рамка должна находиться внутри исходного изображения и иметь ненулевые ширину и высоту.'); return; }
    historyGeneration.current++; uploadAttempt.current = true;
    rememberedRunId.current = null; saveRunSelection({ runId: null, pendingCount: files.length });
    setBusy(true); setUploadingCount(files.length); setUploadError(''); setRun(null); setQueryIndex(0); setSelectedRank(1); setError(''); setRequestId(''); const signal = session.current.signal;
    const body = new FormData(); body.append('gallery_id', galleryId);
    if (inputMode === 'single') { const file = files[0]; const id = file.name.replace(/\.[^.]+$/, ''); const text = `image_id,x,y,w,h\n${csvCell(id)},${bbox.join(',')}\n`; body.append('csv', new File([text], 'query.csv', { type: 'text/csv' })); body.append('images', file); }
    else { body.append('csv', csv[0]); files.forEach((file) => body.append('images', file)); }
    try { const next = await request<Run>(`${baseUrl}/runs`, { method: 'POST', body, signal }); if (!signal.aborted) acceptRun(next); } catch (err) { if (!signal.aborted) { setUploadError(errorMessage(err)); showError(err); } } finally { if (!signal.aborted) setBusy(false); }
  }
  async function cancelRun() { if (!run) return; const id = run.id; const signal = session.current.signal; setBusy(true); try { const next = await request<Run>(`${baseUrl}/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST', signal }); if (!signal.aborted) setRun((current) => current?.id === id ? next : current); } catch (err) { if (!signal.aborted) showError(err); } finally { if (!signal.aborted) setBusy(false); } }
  async function openHistory(id: string) { const signal = session.current.signal; const generation = ++historyGeneration.current; try { const next = await request<Run>(`${baseUrl}/runs/${encodeURIComponent(id)}`, { signal }); if (!signal.aborted && generation === historyGeneration.current) acceptRun(next); } catch (err) { if (!signal.aborted && generation === historyGeneration.current) showError(err); } }
  async function selectFiles(next: File[]) {
    formTouched.current = true;
    const generation = ++fileGeneration.current;
    if (files.length) clearSingleCSV();
    setImageSize([0, 0]); setFiles([]);
    if (inputMode === 'single' && next[0]) {
      if (status && next[0].size > status.limits.max_file_mb * 1024 * 1024) { setError(`Файл превышает лимит ${status.limits.max_file_mb} МБ.`); return; }
      try { const orientation = await imageOrientation(next[0]); if (generation !== fileGeneration.current) return; if (orientation !== 1) { setError('У этого изображения поворот задан метаданными EXIF. Сохраните поворот в самих пикселях, удалив метаданные ориентации, и загрузите снова: иначе рамка не совпадёт с исходными координатами сервера.'); return; } } catch (err) { if (generation === fileGeneration.current) setError(errorMessage(err)); return; }
    }
    if (generation === fileGeneration.current) setFiles(next);
  }
  const fileReady = inputMode === 'single' ? files.length === 1 && imageSize[0] > 0 && !csvReading && !singleCsvIssue : files.length > 0 && csv.length > 0;

  return <div className="app-shell">
    <a className="skip-link" href="#main">К содержимому</a>
    <header className="app-header"><button type="button" className="brand" aria-label="На страницу поиска" onClick={() => setTab('search')}><span className="brand-mark"><ScanLine size={24} strokeWidth={1.7} /></span></button>
      <nav className="primary-nav" aria-label="Разделы стенда"><button className={tab === 'search' ? 'active' : ''} aria-current={tab === 'search' ? 'page' : undefined} onClick={() => setTab('search')} type="button"><Search size={17} /><span>Поиск</span></button><button className={tab === 'report' ? 'active' : ''} aria-current={tab === 'report' ? 'page' : undefined} onClick={() => setTab('report')} type="button"><ShieldCheck size={17} /><span>Проверка решения</span></button></nav>
      <div className="header-actions"><label className="theme-control" title="Цветовая тема">{theme === 'dark' ? <Moon size={17} /> : theme === 'light' ? <Sun size={17} /> : <Monitor size={17} />}<select aria-label="Тема оформления" value={theme} onChange={(event) => setTheme(event.target.value as Theme)}><option value="system">Системная</option><option value="light">Светлая</option><option value="dark">Тёмная</option></select></label><button className="icon-button settings-trigger" type="button" aria-label="Настройки" title="Настройки" aria-haspopup="dialog" onClick={() => setSettingsModal(true)}><Settings size={22} /></button><StatusNotice message={connectionError || (status && !status.available ? status.reason || "Модель пока не подключена." : null)} openRequest={noticeRequest} /></div>
    </header>
    <main id="main" className="main-content">
      {error && <Alert onClose={() => { setError(''); setRequestId(''); }}>{error}{requestId && <small className="request-id">Код запроса: {requestId}</small>}</Alert>}
      {tab === 'report' ? <Report key={`${run?.id || 'pending'}:${historyGeneration.current}`} run={run} galleries={galleries} connected={!!status} uploadingCount={uploadingCount} uploadError={uploadError} onRefresh={() => setRefreshKey((value) => value + 1)} /> : <>
        <div className="page-heading"><div><span className="eyebrow">ВИЗУАЛЬНЫЙ ПОИСК</span><h1>Найдите тот же автомобиль</h1><p>Сопоставление по внешнему виду — без использования госномера.</p></div>{!!run?.results.length && <div className="view-control" role="group" aria-label="Вид изображений"><button type="button" className={cropped ? 'selected' : ''} aria-pressed={cropped} onClick={() => setCropped(true)}><ScanLine size={15} /> Только автомобиль</button><button type="button" className={!cropped ? 'selected' : ''} aria-pressed={!cropped} onClick={() => setCropped(false)}><PanelTop size={15} /> Весь снимок</button></div>}</div>
        <div className="workspace"><aside className="search-rail"><section className="panel search-panel"><div className="section-heading"><h2>Новый поиск</h2><Search size={17} /></div>
          <label className="field-label">Искать в галерее<div className="gallery-select"><select value={galleryId} aria-label="Галерея для поиска" onChange={(event) => { galleryChosen.current = true; setGalleryId(event.target.value); }} disabled={!status || busy || active || deviceChanging || !galleries.length}><option value="" disabled>{!status ? 'Список галерей недоступен' : galleries.length ? 'Выберите галерею' : 'Нет доступных галерей'}</option>{galleries.map((item) => <option value={item.id} key={item.id}>{item.name} · {item.count}</option>)}</select><button type="button" className="icon-button" title="Добавить галерею" aria-label="Добавить галерею" onClick={() => setGalleryModal(true)} disabled={!status?.available || active || busy || deviceChanging}><Plus size={18} /></button></div></label>
          {status && gallery && <p className={`gallery-state ${gallery.status === 'failed' ? 'danger-text' : ''}`}>{gallery.status === 'ready' ? <><span className="status-dot success" />{gallery.count} изображений · готова</> : gallery.status === 'indexing' ? <><LoaderCircle className="spin" size={13} />Индексация: {gallery.processed} / {gallery.count}</> : <>{gallery.error || 'Ошибка индексации'}</>}</p>}

          <form onSubmit={startRun}><div className="segmented" aria-label="Способ загрузки"><button type="button" className={inputMode === 'single' ? 'selected' : ''} onClick={() => changeInputMode('single')} aria-pressed={inputMode === 'single'} disabled={busy || active}><FileImage size={15} /> Один снимок</button><button type="button" className={inputMode === 'batch' ? 'selected' : ''} onClick={() => changeInputMode('batch')} aria-pressed={inputMode === 'batch'} disabled={busy || active}><Layers3 size={15} /> Пакет</button></div>
            {active && !files.length && <p className="field-hint" role="status">Запросы уже загружены на сервер: {run.total} снимков. Ниже показан выполняемый поиск; повторно выбирать файлы не нужно.</p>}
            {inputMode === 'batch' && !active && <p className="field-hint">Выберите несколько JPEG/PNG и файл с координатами автомобилей. Снимки обрабатываются по очереди.</p>}
            <UploadField key={inputMode} label={inputMode === 'single' ? 'Загрузить снимок' : 'Загрузить снимки пакета'} accept="image/jpeg,image/png" multiple={inputMode === 'batch'} files={files} onChange={(next) => void selectFiles(next)} />
            {inputMode === 'single' && <div className="single-coordinates">
              <p className="field-hint">Рамку можно загрузить из CSV или выделить на снимке мышью.</p>
              <UploadField label="Файл с координатами автомобиля (.csv, необязательно)" accept=".csv,text/csv" files={singleCSV ? [singleCSV] : []} onChange={(next) => void selectSingleCSV(next)} />
              <CsvHelp />
              {singleCSV && <button className="text-button" type="button" onClick={clearSingleCSV}>Убрать CSV, задать рамку вручную</button>}
              {singleCsvIssue ? <p className="field-hint danger-text" role="alert">{singleCsvIssue}</p> : singleCSV && <p className="field-hint" role="status">{csvReading ? 'Читаем координаты…' : !files[0] || !imageSize[0] ? `Загрузите снимок «${csvRow?.imageId || ''}».` : 'Рамка загружена из CSV. Её можно поправить мышью или в полях ниже.'}</p>}
            </div>}
            {inputMode === 'single' && files[0] && <BBoxEditor key={`${files[0].name}-${files[0].lastModified}-${files[0].size}`} file={files[0]} bbox={bbox} setBBox={editBBox} onSize={(w, h) => { setBBox((current) => bboxAfterImageLoad(current, imageSize, [w, h])); setImageSize([w, h]); if (!w) setError('Изображение не удалось открыть. Выберите JPEG или PNG.'); }} />}
            {inputMode === 'batch' && <><UploadField label="Файл с координатами автомобилей (.csv)" accept=".csv,text/csv" files={csv} onChange={setCSV} /><CsvHelp /></>}
            <button className="button button-primary search-submit" type="submit" disabled={!ready || !fileReady || busy || active}>{busy ? <LoaderCircle className="spin" size={18} /> : <Search size={18} />}{busy ? 'Загрузка…' : inputMode === 'single' ? 'Найти автомобиль' : 'Обработать пакет'}</button>
            {!status?.available && <p className="search-unavailable">{loading ? "Подключение к серверу…" : <button className="text-button" type="button" onClick={() => setNoticeRequest((value) => value + 1)}>Почему поиск недоступен?</button>}</p>}
            {status?.limits && <p className="limit-note">До {status.limits.max_file_mb} МБ на фото · до {status.limits.max_images} фото</p>}
          </form>
        </section>
        <section className="panel history-panel"><div className="section-heading"><h2>Последние запуски</h2><Clock3 size={16} /></div>{!status ? <p className="muted history-empty">История недоступна без связи с сервером.</p> : history.length ? <div className="history-list">{history.slice(0, 6).map((item) => <button type="button" key={item.id} disabled={busy} onClick={() => void openHistory(item.id)} className={run?.id === item.id ? 'selected' : ''}><span><strong>{item.total} {item.total === 1 ? 'снимок' : 'снимков'}</strong><small>{runLabels[item.status]} · {date(item.created_at)}</small></span><ChevronRight size={16} /></button>)}</div> : <p className="muted history-empty">Здесь появятся выполненные запуски.</p>}</section></aside>
        <div className="results-workspace">{connectionError && run && <Alert kind="warning">Показан последний полученный результат. Состояние запуска не обновляется без связи с сервером.</Alert>}<Results key={run?.id || 'empty'} run={run} queryIndex={queryIndex} setQueryIndex={setQueryIndex} selectedRank={selectedRank} setSelectedRank={setSelectedRank} cropped={cropped} onZoom={(image, title) => setZoom({ image, title })} />
          {run && <div className="run-actions"><span className="muted">{run.export_ready ? 'Полный результат и доказательства готовы.' : active ? 'Можно продолжать просматривать готовые снимки.' : 'Полный экспорт доступен только для успешного запуска.'}</span>{active && <button className="button button-secondary" type="button" disabled={busy} onClick={() => void cancelRun()}><Square size={13} /> Остановить</button>}{run.export_ready && run.artifacts_url && <a className="button button-primary" href={run.artifacts_url} download><ArrowDownToLine size={17} /> Скачать результат</a>}</div>}
        </div></div>
      </>}
    </main>
    {zoom && <Modal title={zoom.title} wide onClose={() => setZoom(null)}><div className="zoom-tools"><span title={zoom.image.image_id}>{zoom.image.image_id}</span><div className="view-control" role="group" aria-label="Вид увеличенного изображения"><button type="button" className={cropped ? 'selected' : ''} aria-pressed={cropped} onClick={() => setCropped(true)}>Только автомобиль</button><button type="button" className={!cropped ? 'selected' : ''} aria-pressed={!cropped} onClick={() => setCropped(false)}>Весь снимок</button></div></div><img className="zoom-image" src={cropped ? zoom.image.crop_url : zoom.image.image_url} alt={zoom.title} /><p className="field-hint">Исходное изображение без цветовых фильтров. Escape — закрыть.</p></Modal>}
    {settingsModal && <Modal title="Настройки" onClose={() => setSettingsModal(false)}><DeviceSettingsForm {...device} blocked={deviceBlocked} onApply={(selected) => void device.apply(selected, deviceBlocked)} onRefresh={device.refresh} /></Modal>}
    {galleryModal && <GalleryModal baseUrl={baseUrl} signal={session.current.signal} disabled={!status?.available || deviceChanging} onClose={() => setGalleryModal(false)} onCreated={(next) => { galleryChosen.current = true; setGalleries((old) => [next, ...old]); setGalleryId(next.id); setGalleryModal(false); }} />}
  </div>;
}
