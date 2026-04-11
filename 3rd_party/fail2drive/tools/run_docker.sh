sudo docker run -it --mount source=/home/jaeger/ordnung/internal/garage_2_cleanup/results/,target=/workspace/results,type=bind --rm --net=host --gpus '"device=0"' -e PORT=2000 docker.io/library/leaderboard-user:latest

