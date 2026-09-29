const example = 'image_id,x,y,w,h\ncar_001,120,80,640,360\n';

export default function CsvHelp() {
  return <details className="csv-help"><summary>Что за файл с координатами?</summary>
    <p>Одна строка указывает автомобиль на одном снимке. Координаты даны в пикселях исходного изображения.</p>
    <dl><div><dt>image_id</dt><dd>Имя файла без расширения</dd></div><div><dt>x, y</dt><dd>Левый верхний угол рамки</dd></div><div><dt>w, h</dt><dd>Ширина и высота рамки</dd></div></dl>
    <p>Пример для <code>car_001.jpg</code>. Замените имя и координаты своими данными.</p>
    <a className="text-link" href={`data:text/csv;charset=utf-8,${encodeURIComponent(example)}`} download="coordinates-example.csv">Скачать пример CSV</a>
  </details>;
}
