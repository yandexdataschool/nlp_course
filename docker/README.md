ROOT_DIR # /home/ragilyazev/projects - директория с проектом
DATA_DIR # /home/ragilyazev/data - директория с данными (большие файлы загружаем отдельно от проекта)

docker-compose build  # собрать имадж
docker compose up -d  # поднять контейнер
docker compose down   # остановить контейнер