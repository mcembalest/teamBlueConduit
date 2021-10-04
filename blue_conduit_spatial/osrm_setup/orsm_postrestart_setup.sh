sudo /bin/dd if=/dev/zero of=/var/swapfile bs=1M count=10240
sudo /sbin/mkswap /var/swapfile
sudo chmod 600 /var/swapfile
sudo /sbin/swapon /var/swapfile


wget http://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf

docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/michigan-latest.osm.pbf
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/michigan-latest.osrm
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/michigan-latest.osrm
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/michigan-latest.osrm

