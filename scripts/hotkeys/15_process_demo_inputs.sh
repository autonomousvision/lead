base="outputs/evaluation/819_thesis_defense/001_before_0/251122_163745/simulator_outputs"

for d in "$base"/*/; do
    bash 17_create_demo_video.sh "$d"
done
