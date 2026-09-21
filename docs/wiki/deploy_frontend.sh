#!/usr/bin/env bash
# Выкладка фронтенда Tellscope и синхронизация документации wiki.
#
# Запускать от root:  sudo bash /opt/tellscope-deploy/deploy_frontend.sh
#
# Что делает по шагам:
#   1) собирает фронтенд (yarn build);
#   2) публикует сборку в /var/www/tellscope и перезапускает nginx;
#   3) обновляет подписи интерфейса (labels.txt) и синхронизирует страницы wiki.
#
# Синхронизация идемпотентна: если тексты не менялись, wiki не трогается.
set -euo pipefail

FE=/home/dev/tellscope_app/tellscope_frontend
BE=/home/dev/tellscope_app/tellscope_backend
WIKI=/opt/tellscope-wiki
PY=$BE/venv_py312_clean/bin/python
WEB=/var/www/tellscope

say() { printf '\n== %s\n' "$1"; }

if [ "$(id -u)" != "0" ]; then
  echo "нужны права root: sudo bash $0" >&2
  exit 1
fi

say "сборка фронтенда"
cd "$FE"
yarn build

say "публикация сборки"
cp -r dist/. "$WEB"/
service nginx restart >/dev/null
BUNDLE=$(grep -o 'index-[A-Za-z0-9_-]*\.js' "$WEB/index.html" | head -1 || true)
echo "выложен бандл: ${BUNDLE:-не найден}"

say "подписи интерфейса для документации"
"$PY" "$WIKI/wiki_labels.py" | tail -1

say "синхронизация wiki"
"$PY" "$WIKI/wiki_sync.py" | tail -5

say "готово"
echo "фронтенд: https://tellscope40.headsmade.com"
echo "документация: https://tellscope40.headsmade.com:8445"
