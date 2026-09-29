import type { ComputeDevice, DeviceSettings } from './contracts';

export const deviceLabels: Record<ComputeDevice, string> = {
  cuda: 'GPU — видеокарта (NVIDIA)',
  cpu: 'CPU — процессор',
};

const isDevice = (value: unknown): value is ComputeDevice => value === 'cuda' || value === 'cpu';
const isNullableText = (value: unknown) => value === null || typeof value === 'string';

export function isDeviceSettings(value: unknown): value is DeviceSettings {
  if (!value || typeof value !== 'object') return false;
  const settings = value as Record<string, unknown>;
  return isDevice(settings.selected_device)
    && (settings.active_device === null || isDevice(settings.active_device))
    && typeof settings.switching === 'boolean' && typeof settings.can_switch === 'boolean'
    && isNullableText(settings.error) && Array.isArray(settings.devices)
    && settings.devices.length === 2
    && new Set(settings.devices.map((device) => device?.id)).size === 2
    && settings.devices.every((device) => device && typeof device === 'object'
      && isDevice(device.id) && typeof device.name === 'string' && device.name.length > 0
      && typeof device.available === 'boolean' && isNullableText(device.reason));
}

export function canApplyDevice(settings: DeviceSettings | null, device: ComputeDevice, blocked = false): boolean {
  return !!settings && !blocked && !settings.switching && settings.can_switch
    && settings.devices.some((option) => option.id === device && option.available)
    && (device !== settings.selected_device || device !== settings.active_device || !!settings.error);
}

// A completed reload may change model fingerprints, gallery indices and quality reports.
export function deviceTransitionNeedsRefresh(previous: DeviceSettings | null, next: DeviceSettings): boolean {
  return !!previous && !next.switching && (previous.switching
    || previous.active_device !== next.active_device || previous.selected_device !== next.selected_device);
}
