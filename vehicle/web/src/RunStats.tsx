import type { Run } from './contracts';

const number = (value: number) => new Intl.NumberFormat('ru', { maximumFractionDigits: 1 }).format(value);

export default function RunStats({ run }: { run: Run }) {
  const timing = run.timing;
  if (!timing) return null;
  return <div className="run-statistics">
    <div className="run-stats" aria-label="Скорость текущего поиска">
      <span>Время поиска <strong>{number(timing.processing_ms / 1000)} с</strong></span>
      <span>Скорость <strong>{timing.images_per_second === null ? 'Ожидаем результат' : `${number(timing.images_per_second)} фото/с`}</strong></span>
    </div>
    <details><summary>Как считается время</summary><p>Обработка снимков и поиск по галерее. Загрузка файлов, ожидание очереди и упаковка результата сюда не входят. Это скорость сервиса, а не конкурсный замер модели.</p>
      <dl><div><dt>В очереди</dt><dd>{number(timing.queue_ms / 1000)} с</dd></div>
        <div><dt>Успешно обработано</dt><dd>{timing.successful_images}</dd></div>
        {timing.mean_image_ms !== null && <div><dt>Среднее на снимок</dt><dd>{number(timing.mean_image_ms)} мс</dd></div>}</dl>
    </details>
  </div>;
}
