# Контроль E27: 77 запросов / 324 снимка галереи

Исходные query.csv, gallery.csv, ground_truth.csv и control-manifest.json не изменены. Фотографии берутся из полного датасета по image_id; рамки — из CSV.

embeddings.npy содержит 401 исходный вектор float32 размерности 384: сначала запросы, затем галерея. Векторы получены на NVIDIA A10 выбранным FP32-извлечением с 4 обработчиками и batch32. submission.csv и candidates.csv сериализованы из сохранённых принятых решений; official_report.json перенесён из строгой CPU-проверки этих GPU-векторов.

Постобработка: gallery4NN, mix0.25, k-reciprocal top50/k1=5/lambda0.6; поддержка3 соседей, alpha0.5, порог0.713592811641558. Порог не подбирался повторно.

Качество: mAP@10 82.139284%; Rank-1 82.258065%; Rank-5 93.548387%; F1 90.265487%; TNR100% (15/15). Исторический контроль ранее изучался; это не независимый holdout и не новый прогон собранного веба/Docker. SHA исходников и производных файлов записаны в source-manifest.json.

Воспроизведение на своём оборудовании после сборки образа:

```sh
docker compose run --rm export --images /data/images --query /app/control/query.csv --gallery /app/control/gallery.csv --output /results/control77 --ground-truth /app/control/ground_truth.csv
```

В .env должны быть указаны существующие каталоги датасета и результатов. Используйте новую папку результата. При отсутствии GPU замените сервис export на export-cpu и добавьте --profile cpu перед run. Эталон используется только официальным оценщиком после поиска.
