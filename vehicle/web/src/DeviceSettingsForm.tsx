import { useEffect, useState, type FormEvent } from 'react';
import { Cpu, LoaderCircle, Monitor } from 'lucide-react';
import type { ComputeDevice, DeviceSettings } from './contracts';
import { canApplyDevice, deviceLabels } from './deviceSettings';

interface Props {
  settings: DeviceSettings | null;
  loading: boolean;
  saving: boolean;
  fetchError: string;
  saveError: string;
  requestId: string;
  blocked: boolean;
  onApply: (device: ComputeDevice) => void;
  onRefresh: () => void;
}

export default function DeviceSettingsForm({ settings, loading, saving, fetchError, saveError, requestId, blocked, onApply, onRefresh }: Props) {
  const [selected, setSelected] = useState<ComputeDevice>(settings?.selected_device || 'cuda');
  useEffect(() => { if (settings) setSelected(settings.selected_device); }, [settings?.selected_device]);
  const switching = saving || !!settings?.switching;
  const disabled = loading || switching || blocked || !!fetchError || !settings?.can_switch;
  function submit(event: FormEvent) {
    event.preventDefault();
    if (canApplyDevice(settings, selected, disabled)) onApply(selected);
  }
  return <form className="device-settings" onSubmit={submit}>
    <fieldset disabled={disabled} aria-describedby="device-reload-hint">
      <legend>Устройство вычислений</legend>
      {(['cuda', 'cpu'] as const).map((id) => {
        const device = settings?.devices.find((option) => option.id === id);
        return <label className={`device-option ${selected === id ? 'is-selected' : ''} ${device && !device.available ? 'is-unavailable' : ''}`} key={id}>
          <input type="radio" name="compute-device" value={id} checked={selected === id} disabled={!device?.available}
            onChange={() => setSelected(id)} aria-describedby={device?.reason ? `device-reason-${id}` : undefined} />
          {id === 'cuda' ? <Monitor size={20} /> : <Cpu size={20} />}
          <span><strong>{deviceLabels[id]}</strong>{id === 'cuda' && <small>{device?.available ? `${device.name} · по умолчанию` : 'По умолчанию'}</small>}
            {device && !device.available && <small id={`device-reason-${id}`}>{device.reason || 'Устройство недоступно на этом сервере.'}</small>}</span>
        </label>;
      })}
    </fieldset>
    <div className="device-current" role="status" aria-live="polite">
      {loading ? <><LoaderCircle size={16} className="spin" /> Загружаем настройки…</> : fetchError ? 'Состояние устройства недоступно.' : switching
        ? <><LoaderCircle size={16} className="spin" /> Перезагрузка модели и индексов…</>
        : settings ? <><span className={`status-dot ${settings.active_device ? 'success' : ''}`} />{settings.active_device ? `Вычисления: ${deviceLabels[settings.active_device]}` : 'Модель не загружена.'}</>
        : 'Состояние устройства недоступно.'}
    </div>
    <p className="field-hint" id="device-reload-hint">Выбор сохраняется на сервере. При применении модель и индексы галерей перезагрузятся; поиск временно будет недоступен.</p>
    {!loading && !switching && !fetchError && (blocked || settings && !settings.can_switch) && <p className="field-hint">Смена устройства станет доступна после завершения текущей обработки или загрузки модели.</p>}
    {settings?.error && <p className="device-error" role="alert">{settings.error}</p>}
    {(fetchError || saveError) && <div className="device-error" role="alert"><p>{fetchError || saveError}</p>{requestId && <small className="request-id">Код запроса: {requestId}</small>}
      {fetchError && <button type="button" className="text-button" onClick={onRefresh} disabled={loading}>Повторить загрузку настроек</button>}</div>}
    <div className="device-settings-actions"><button className="button button-primary" type="submit" disabled={!canApplyDevice(settings, selected, disabled)}>
      {switching && <LoaderCircle size={17} className="spin" />}{switching ? 'Перезагрузка…' : 'Применить'}</button></div>
  </form>;
}
