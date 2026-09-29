import { useEffect, useRef, useState } from 'react';
import { errorMessage, request, ServiceError } from './api';
import type { ComputeDevice, DeviceSettings } from './contracts';
import { canApplyDevice, deviceTransitionNeedsRefresh, isDeviceSettings } from './deviceSettings';

export default function useDeviceSettings(baseUrl: string, onChanged: () => void) {
  const [settings, setSettings] = useState<DeviceSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [fetchError, setFetchError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [requestId, setRequestId] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);
  const current = useRef<DeviceSettings | null>(null);
  const savingRef = useRef(false);
  const generation = useRef(0);
  const session = useRef(new AbortController());
  const changed = useRef(onChanged); changed.current = onChanged;

  useEffect(() => {
    const controller = new AbortController(); session.current = controller;
    return () => controller.abort();
  }, []);

  function accept(value: unknown) {
    if (!isDeviceSettings(value)) throw new ServiceError('Настройки сервера имеют неизвестный формат.', 'INVALID_RESPONSE');
    const refresh = deviceTransitionNeedsRefresh(current.current, value);
    current.current = value; setSettings(value); setFetchError('');
    if (refresh) changed.current();
    return value;
  }

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      const version = generation.current;
      try {
        if (savingRef.current) return;
        const value = await request<unknown>(`${baseUrl}/settings`, { signal: controller.signal });
        if (!controller.signal.aborted && generation.current === version && !savingRef.current) accept(value);
      } catch (error) {
        if (!controller.signal.aborted && generation.current === version) setFetchError(errorMessage(error));
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
          timer = window.setTimeout(() => void poll(), current.current?.switching ? 1000 : 4000);
        }
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [baseUrl, refreshKey]);

  async function apply(device: ComputeDevice, blocked = false) {
    if (savingRef.current || fetchError || !canApplyDevice(current.current, device, blocked)) return;
    const signal = session.current.signal;
    generation.current++; savingRef.current = true; setSaving(true); setSaveError(''); setRequestId('');
    try {
      const value = await request<unknown>(`${baseUrl}/settings/device`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device }), signal,
      });
      if (!signal.aborted) { accept(value); changed.current(); }
    } catch (error) {
      if (!signal.aborted) {
        setSaveError(errorMessage(error)); setRequestId(error instanceof ServiceError ? error.requestId : '');
      }
    } finally {
      savingRef.current = false;
      if (!signal.aborted) { setSaving(false); setRefreshKey((value) => value + 1); }
    }
  }

  return { settings, loading, saving, fetchError, saveError, requestId, apply,
    refresh: () => { setLoading(true); setRefreshKey((value) => value + 1); } };
}
