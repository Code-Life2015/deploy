#!/bin/bash
# #param  dns name, $1 ca_tls name.
param=$2
if [ -z $param ]; then
  param=$1
fi
sudo openssl genrsa -out $1.key 4096
sudo openssl req -sha512 -new   -subj "/C=CN/ST=Beijing/L=Beijing/O=example/OU=Personal/CN=${param}" -key $1.key -out $1.csr
sudo cat > v4.ext <<-EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1=$param
DNS.2=$(hostname)
DNS.3=localhost
IP.1=127.0.0.1
EOF
sudo openssl x509 -req -sha512 -days 3650 -extfile v4.ext -CA ca1.crt -CAkey ca1.key -CAcreateserial -in $1.csr -out $1.crt
