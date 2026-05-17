# Эта папка намеренно пустая в репозитории.
# Веса backbone-ов скачиваются автоматически скриптом scripts/download_weights.py
# во время сборки EXE на CI (см. .github/workflows/build-exe.yml).
#
# Для локальной разработки интернет не нужен если вы уже запускали приложение:
# torchvision кеширует скачанные веса в ~/.cache/torch/hub/checkpoints/
# и feature_extractor.py находит их автоматически.
#
# Чтобы собрать EXE локально с offline-весами:
#   python scripts/download_weights.py
