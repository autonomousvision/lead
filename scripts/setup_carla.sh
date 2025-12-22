mkdir -p 3rd_party/CARLA_0915

cd 3rd_party/CARLA_0915

wget -O CARLA_0915.tar.gz https://tiny.carla.org/carla-0-9-15-linux
tar -xvzf CARLA_0915.tar.gz
cd Import
wget -O AdditionalMaps_0.9.15.tar.gz https://tiny.carla.org/additional-maps-0-9-15-linux
tar -xvzf AdditionalMaps_0.9.15.tar.gz
cd ..
bash ImportAssets.sh
