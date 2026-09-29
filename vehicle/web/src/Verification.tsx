import { useEffect, useRef, useState } from 'react';
import { ArrowDownToLine, FileSpreadsheet, LoaderCircle, RefreshCw } from 'lucide-react';
import { errorMessage, request, ServiceError } from './api';
import type { EvaluationReport, Gallery, Run, RunEvaluation } from './contracts';

const ACTIVE = new Set(['queued', 'running']);
const EMPTY: RunEvaluation = { status: 'unavailable', report: null, error: null, ground_truth_sha256: null };
const number = (value: number) => new Intl.NumberFormat('ru', { maximumFractionDigits: 1 }).format(value);
const rate = (value: number) => new Intl.NumberFormat('ru', { maximumFractionDigits: 2 }).format(value);
const metrics = [
  ['map_at_10', 'mAP@10', 'Качество первых 10 результатов'],
  ['rank_1', 'Rank-1', 'Верная машина на первом месте'],
  ['rank_5', 'Rank-5', 'Верная машина среди первых пяти'],
  ['micro_f1', 'F1', 'Баланс точности и полноты совпадений'],
  ['tnr', 'TNR', 'Правильные отказы при отсутствии пары'],
] as const;

export function qualityReportForRun(run: Run | null, evaluation: RunEvaluation): EvaluationReport | null {
  const report = evaluation.report;
  if (!run || run.status !== 'completed' || evaluation.status !== 'completed' || !report?.measured || !report.metrics
    || report.run_id !== run.id || !report.model?.sha256 || !run.model || report.model.sha256 !== run.model.sha256
    || !evaluation.ground_truth_sha256 || report.ground_truth_sha256 !== evaluation.ground_truth_sha256
    || (run.model.inference_fingerprint && report.model.inference_fingerprint !== run.model.inference_fingerprint)) return null;
  return report;
}

export function reportVisibilityKey(report: EvaluationReport | null) {
  if (!report?.run_id || !report.model?.inference_fingerprint || !report.calibration_sha256) return null;
  return `vehicle-hidden-run-report:${JSON.stringify([report.run_id, report.model.inference_fingerprint, report.calibration_sha256])}`;
}

export function reportIsHidden(report: EvaluationReport | null, storage?: Pick<Storage, 'getItem'>) {
  const key = reportVisibilityKey(report);
  try { return !!key && (storage || sessionStorage).getItem(key) === '1'; } catch { return false; }
}

export function setReportHidden(report: EvaluationReport | null, hidden: boolean, storage?: Pick<Storage, 'setItem'>) {
  const key = reportVisibilityKey(report);
  if (key) { try { (storage || sessionStorage).setItem(key, hidden ? '1' : '0'); } catch { /* Current view still updates when browser storage is unavailable. */ } }
  return key;
}

export function isRunEvaluation(value: unknown): value is RunEvaluation {
  if (!value || typeof value !== 'object') return false;
  const state = value as RunEvaluation;
  return ['unavailable', 'waiting', 'running', 'completed', 'failed'].includes(state.status)
    && (state.error === null || typeof state.error === 'string')
    && (state.ground_truth_sha256 === null || typeof state.ground_truth_sha256 === 'string')
    && (state.report === null || typeof state.report === 'object');
}

export function VerificationSummary({ run, galleries, connected, evaluation, uploadingCount = null, uploadError = '', hidden = false, evaluationUploading = false }: {
  run: Run | null; galleries: Gallery[]; connected: boolean; evaluation: RunEvaluation;
  uploadingCount?: number | null; uploadError?: string; hidden?: boolean; evaluationUploading?: boolean;
}) {
  const selected = uploadingCount === null ? run : null;
  const gallery = galleries.find((item) => item.status === 'indexing') || galleries[0];
  const report = qualityReportForRun(selected, evaluation);
  const quality = hidden ? null : report;
  const stopped = selected && ['failed', 'cancelled'].includes(selected.status);
  const stage = uploadingCount !== null || !selected ? 0 : ACTIVE.has(selected.status) || stopped ? 1 : report ? 3 : 2;
  const matched = selected?.results.filter((item) => item.status === 'matched').length;
  const rejected = selected?.results.filter((item) => item.status === 'rejected').length;
  const timing = selected?.timing;
  let qualityNote = 'Нет эталонных ответов. Без них точность этого запуска рассчитать нельзя.';
  if (uploadingCount !== null) qualityNote = uploadError ? 'Новый запуск не подтверждён. Метрики предыдущих запусков здесь не показываются.' : 'Новый пакет ещё принимается. Показатели появятся после обработки и оценки этого запуска.';
  else if (!selected) qualityNote = 'Выберите запуск в истории или выполните новый поиск.';
  else if (stopped) qualityNote = 'Поиск не завершён успешно. Для расчёта качества нужен завершённый результат.';
  else if (evaluationUploading) qualityNote = 'Проверяем новые эталонные ответы. Предыдущая оценка скрыта до подтверждения этой попытки.';
  else if (evaluation.status === 'waiting') qualityNote = 'Эталонные ответы прикреплены. Качество рассчитается автоматически после завершения поиска.';
  else if (evaluation.status === 'running') qualityNote = 'Официальный оценщик рассчитывает качество этого запуска.';
  else if (evaluation.status === 'failed') qualityNote = `Не удалось рассчитать качество: ${evaluation.error || 'подробности отсутствуют'}`;
  else if (evaluation.status === 'completed' && !report) qualityNote = 'Отчёт отклонён: он не подтверждает этот запуск и эту модель.';
  else if (report) qualityNote = hidden ? 'Отчёт этого запуска скрыт. Его можно снова показать; файлы сохранены.' : 'Рассчитано официальным оценщиком по результатам и эталонным ответам именно этого запуска.';
  else if (ACTIVE.has(selected.status)) qualityNote += ' Их можно прикрепить сейчас; оценка начнётся после поиска.';
  return <>
    <ol className="verification-stages" aria-label="Этапы проверки">{['Загрузка', 'Поиск', 'Оценка качества', 'Готово'].map((label, index) => <li key={label} className={index === stage ? stopped || evaluation.status === 'failed' || uploadError ? 'current danger-text' : 'current' : index < stage ? 'done' : ''} aria-current={index === stage ? 'step' : undefined}><span>{index + 1}</span>{label}</li>)}</ol>
    <section className="panel current-verification" aria-label="Выбранный запуск">
      {!connected && <p className="verification-warning" role="status">Нет связи с сервером. Показано последнее полученное состояние.</p>}
      {uploadingCount !== null ? <>
        <h2>{uploadError ? 'Ошибка новой загрузки' : uploadingCount === 1 ? 'Загрузка нового снимка' : 'Загрузка нового пакета'}</h2>
        <p role="status">{uploadingCount} {uploadingCount === 1 ? 'снимок' : 'снимков'} с рамками автомобилей.</p>
        {uploadError ? <><p role="alert" className="danger-text">{uploadError}</p><p className="field-hint">Подтверждение запуска не получено. Перед повторной отправкой проверьте историю: при обрыве связи сервер мог принять файлы.</p></> : <><progress aria-label="Приём и проверка файлов" /><p className="field-hint">Сервер принимает файлы и проверяет рамки. Процент передачи недоступен. Не обновляйте страницу до подтверждения.</p></>}
      </> : selected ? <>
        <div className="section-heading"><h2>{ACTIVE.has(selected.status) ? 'Поиск выполняется' : selected.status === 'completed' ? 'Поиск завершён' : selected.status === 'cancelled' ? 'Поиск отменён' : 'Ошибка поиска'}</h2><span>{selected.model?.device || 'Устройство не указано'}</span></div>
        <p className="verification-run-id">Запуск: <code>{selected.id}</code></p>
        <div className="verification-progress"><strong>{selected.processed} из {selected.total} снимков</strong><progress aria-label="Прогресс выбранного запуска" value={selected.processed} max={Math.max(1, selected.total)} /></div>
        {selected.error && <p className="danger-text" role="alert">{selected.error}</p>}
      </> : <><h2>{gallery?.status === 'indexing' ? 'Индексация галереи' : 'Запуск не выбран'}</h2>{gallery?.status === 'indexing' ? <div className="verification-progress"><strong>{gallery.processed} из {gallery.count} снимков</strong><progress aria-label="Прогресс индексации" value={gallery.processed} max={Math.max(1, gallery.count)} /></div> : <p className="field-hint">Выполните поиск или выберите завершённый запуск в истории. Фотографии галереи готовятся к поиску отдельно.</p>}</>}
      <dl className="verification-live" aria-label="Показатели этого запуска">
        <div><dt>Время поиска</dt><dd>{timing ? `${number(timing.processing_ms / 1000)} с` : '—'}</dd></div>
        <div><dt>Скорость</dt><dd>{timing?.images_per_second != null ? `${rate(timing.images_per_second)} фото/с` : '—'}</dd></div>
        <div><dt>Совпадения приняты</dt><dd>{matched ?? '—'}</dd></div>
        <div><dt>Отказы</dt><dd>{rejected ?? '—'}</dd></div>
        <div><dt>Ошибки обработки</dt><dd>{selected?.failed ?? '—'}</dd></div>
      </dl>
      <p className="field-hint">Принятия и отказы — решения модели, а не число правильных ответов. Время относится к обработке и поиску, без загрузки файлов, очереди и упаковки; это не конкурсный замер GPU.</p>
      {selected && <div className="report-actions"><a className="text-link" href={selected.evidence_url} target="_blank" rel="noreferrer"><ArrowDownToLine size={15} /> Журнал этого запуска</a>{selected.export_ready && selected.artifacts_url && <a className="button button-secondary" href={selected.artifacts_url} download>Скачать результат поиска</a>}</div>}
    </section>
    <section className="verification-quality" aria-label="Качество этого запуска"><h2>Качество этого запуска</h2><p className={`field-hint ${evaluation.status === 'failed' && selected ? 'danger-text' : ''}`} role="status">{qualityNote}</p>
      <div className="metrics-grid">{metrics.map(([key, title, explanation]) => {
        const value = quality?.metrics?.[key];
        const measured = typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
        return <div className={`metric-card ${measured ? '' : 'metric-pending'}`} key={key}><span>{title}</span><strong>{measured ? `${number(value * 100)}%` : '—'}</strong><p>{explanation}</p></div>;
      })}</div>
      {quality && <><p className="control-dataset"><strong>Выборка этого запуска:</strong> {quality.dataset || 'Указана в отчёте'}</p><details className="technical-details"><summary>Основание оценки и ограничения</summary><p>Запуск оценки: <code>{quality.run_id}</code></p>{quality.notes.length > 0 && <ul>{quality.notes.map((note) => <li key={note}>{note}</li>)}</ul>}</details></>}
    </section>
  </>;
}

export default function Verification({ run, galleries, connected, uploadingCount = null, uploadError = '', onRefresh }: {
  run: Run | null; galleries: Gallery[]; connected: boolean; uploadingCount?: number | null; uploadError?: string; onRefresh: () => void;
}) {
  const current = uploadingCount === null ? run : null;
  const id = current?.id || null;
  const [local, setLocal] = useState<{ id: string | null; state: RunEvaluation }>({ id, state: current?.evaluation || EMPTY });
  const [file, setFile] = useState<File | null>(null);
  const [sending, setSending] = useState(false);
  const [attachError, setAttachError] = useState('');
  const [fetchError, setFetchError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [hiddenOverride, setHiddenOverride] = useState<{ key: string | null; value: boolean } | null>(null);
  const generation = useRef(0);
  const postPending = useRef(false);
  const session = useRef<AbortController | null>(null);
  useEffect(() => { setFile(null); setAttachError(''); setSending(false); setHiddenOverride(null); }, [id]);
  useEffect(() => {
    const controller = new AbortController(); session.current = controller; generation.current++; postPending.current = false;
    setLocal((old) => old.id === id ? old : { id, state: current?.evaluation || EMPTY });
    setFetchError('');
    let timer: number;
    async function poll() {
      if (postPending.current) { timer = window.setTimeout(() => void poll(), 2500); return; }
      const requestGeneration = generation.current;
      try {
        const state = await request<RunEvaluation>(`/api/v1/runs/${encodeURIComponent(id!)}/evaluation`, { signal: controller.signal });
        if (!isRunEvaluation(state)) throw new ServiceError('Состояние оценки имеет неизвестный формат.', 'INVALID_RESPONSE');
        if (!controller.signal.aborted && requestGeneration === generation.current) { setLocal({ id, state }); setFetchError(''); }
      } catch (error) { if (!controller.signal.aborted && requestGeneration === generation.current) setFetchError(errorMessage(error)); }
      finally { if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), 2500); }
    }
    if (id) void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [id, refresh]);
  const evaluation = local.id === id ? local.state : current?.evaluation || EMPTY;
  const displayEvaluation = attachError ? { ...EMPTY, status: 'failed' as const, error: attachError } : sending ? EMPTY : evaluation;
  const report = qualityReportForRun(current, displayEvaluation);
  const key = reportVisibilityKey(report);
  const hidden = hiddenOverride?.key === key ? hiddenOverride.value : reportIsHidden(report);
  const evaluating = evaluation.status === 'waiting' || evaluation.status === 'running';
  const mayAttach = !!current && ['queued', 'running', 'completed'].includes(current.status);
  async function attach(event: React.FormEvent) {
    event.preventDefault();
    const controller = session.current;
    if (!id || !file || !mayAttach || evaluating || sending || !controller || controller.signal.aborted) return;
    if (file.size > 1024 * 1024) { setAttachError('CSV должен быть не больше 1 МиБ.'); return; }
    const requestGeneration = ++generation.current;
    postPending.current = true;
    setReportHidden(report, false); setHiddenOverride(null);
    setSending(true); setAttachError('');
    const data = new FormData(); data.append('ground_truth', file);
    try {
      const state = await request<RunEvaluation>(`/api/v1/runs/${encodeURIComponent(id)}/evaluation`, { method: 'POST', body: data, signal: controller.signal });
      if (!isRunEvaluation(state)) throw new ServiceError('Состояние оценки имеет неизвестный формат.', 'INVALID_RESPONSE');
      if (!controller.signal.aborted && requestGeneration === generation.current) {
        generation.current++; postPending.current = false;
        setLocal({ id, state }); setHiddenOverride(null); setFile(null); setSending(false);
      }
    } catch (error) {
      if (!controller.signal.aborted && requestGeneration === generation.current) {
        generation.current++; postPending.current = false; setAttachError(errorMessage(error)); setSending(false);
      }
    }
  }
  function toggleReport() { const next = !hidden; setReportHidden(report, next); setHiddenOverride({ key, value: next }); }
  return <div className="report-page">
    <div className="page-heading"><div><span className="eyebrow">ОДИН ЗАПУСК · ОДИН ОТЧЁТ</span><h1>Проверка решения</h1><p>Ход поиска и проверка качества выбранного запуска.</p></div><button className="button button-secondary" type="button" disabled={sending} onClick={() => { setRefresh((value) => value + 1); onRefresh(); }}><RefreshCw size={16} /> Обновить</button></div>
    <VerificationSummary run={current} galleries={galleries} connected={connected} evaluation={displayEvaluation} uploadingCount={uploadingCount} uploadError={uploadError} hidden={hidden} evaluationUploading={sending} />
    {current && <section className="panel verification-labels"><div className="section-heading"><h2>Эталонные ответы</h2><FileSpreadsheet size={20} /></div><p className="field-hint">Для оценки нужны правильные ответы для запросов и галереи этого запуска. Они используются только оценщиком; нейросеть их не получает. Обычный CSV с рамками не подходит.</p>
      <form onSubmit={attach}><label className="field-label">Эталонные ответы (CSV)<input aria-label="Эталонные ответы (CSV)" type="file" accept=".csv,text/csv" disabled={!mayAttach || evaluating || sending} onChange={(event) => { setFile(event.target.files?.[0] || null); setAttachError(''); }} /></label><button className="button button-secondary" type="submit" disabled={!mayAttach || evaluating || sending || !file}>{sending ? <LoaderCircle size={16} className="spin" /> : <FileSpreadsheet size={16} />}{sending ? 'Проверяем эталон…' : current.status === 'completed' ? 'Рассчитать качество' : 'Прикрепить эталон'}</button></form>
      <details className="csv-help"><summary>Каким должен быть файл</summary><p>CSV в UTF-8, до 1 МиБ. Поля: <code>image_id,vehicle_id,camera_id,split</code>. <code>split</code> — <code>query</code> или <code>gallery</code>. ID должны в точности соответствовать изображениям этого запуска.</p><p>Эталон можно прикрепить во время поиска: оценка начнётся после его завершения. Для официальных тестовых снимков без опубликованных ответов доступен результат поиска, но не точность.</p></details>
      {attachError && <p className="danger-text" role="alert">{attachError}</p>}{fetchError && <p className="verification-warning" role="status">Не удалось обновить состояние оценки: {fetchError}</p>}
      {report && <div className="report-actions"><button className="text-button" type="button" onClick={toggleReport}>{hidden ? 'Показать отчёт этого запуска' : 'Скрыть отчёт этого запуска'}</button><a className="text-link" href={`/api/v1/runs/${encodeURIComponent(current.id)}/evaluation/report`} download><ArrowDownToLine size={16} /> Скачать отчёт качества (JSON)</a></div>}
    </section>}
  </div>;
}
