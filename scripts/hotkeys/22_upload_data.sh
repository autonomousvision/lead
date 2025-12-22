src_root="data/carla_leaderboad2_v12/results/data/sensor_data/zip/"
dst_root="/mnt/lustre/work/geiger/gwb581/lead/data/carla_leaderboad2_v12/results/data/sensor_data/zip/"
rsync -avz --info=progress2 $src_root cluster:$dst_root
