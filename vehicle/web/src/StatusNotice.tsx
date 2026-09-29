import { useEffect, useState } from 'react';
import { Bell, TriangleAlert, X } from 'lucide-react';

export default function StatusNotice({ message, openRequest }: { message: string | null; openRequest: number }) {
  const [automatic, setAutomatic] = useState(false);
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    setOpened(false);
    setAutomatic(Boolean(message));
    if (!message) return;
    const timer = window.setTimeout(() => setAutomatic(false), 15_000);
    return () => window.clearTimeout(timer);
  }, [message]);
  useEffect(() => { if (openRequest > 0) setOpened(true); }, [openRequest]);
  const visible = opened || Boolean(message && automatic);
  function close() { setOpened(false); setAutomatic(false); }
  return <div className="status-notice" onKeyDown={(event) => { if (event.key === 'Escape') close(); }}>
    <button className={`icon-button notification-trigger ${message ? 'has-warning' : ''}`} type="button"
      aria-label={message ? 'Уведомления: поиск недоступен' : 'Уведомления: нет предупреждений'}
      aria-expanded={visible} aria-controls="service-notification" onClick={() => visible ? close() : setOpened(true)}>
      {message ? <TriangleAlert size={23} /> : <Bell size={22} />}
    </button>
    <div id="service-notification" className={`notification-panel ${visible ? 'is-visible' : ''}`}
      aria-hidden={!visible} inert={!visible}>
      <div className="notification-heading"><strong>{message ? 'Поиск недоступен' : 'Уведомления'}</strong>
        <button className="icon-button" type="button" aria-label="Скрыть уведомление" onClick={close}><X size={18} /></button></div>
      <p role="status">{message || 'Нет активных предупреждений.'}</p>
    </div>
  </div>;
}
