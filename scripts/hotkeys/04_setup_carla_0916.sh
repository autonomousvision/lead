mkdir -p 3rd_party/CARLA_0916

cd 3rd_party/CARLA_0916

wget -O CARLA_0916.tar.gz https://tiny.carla.org/carla-0-9-16-linux
tar -xvzf CARLA_0916.tar.gz
cd Import
wget -O AdditionalMaps_0.9.16.tar.gz https://tiny.carla.org/additional-maps-0-9-16-linux
tar -xvzf AdditionalMaps_0.9.16.tar.gz
cd ..
bash ImportAssets.sh
