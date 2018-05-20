#!/bin/bash

# only works on ubuntu.
# need to pass host IP


export HostIP=$1
echo "Host IP is $HostIP"


docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd-naoshima quay.io/coreos/etcd:v2.3.8 \
 -name etcd0 \
 -advertise-client-urls http://${HostIP}:2379,http://${HostIP}:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://${HostIP}:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://${HostIP}:2380 \
 -initial-cluster-state new


sleep 1  # give it time to start

echo 'Setting key x = 3'
etcdctl --debug set x 3


echo 'Getting key x'
etcdctl --debug get x


echo 'Killing container'
docker kill etcd-naoshima
docker rm etcd-naoshima
