
#!/bin/sh

#########################################################
# version: 0.1 beta
# ###Copy  by ture2###############
# Requirement: You need to have Debian or UBUNTU.
# home dir#############################################

pwddir=$(pwd)
username=$(whoami)
groupname=$username
hostname1=reg.hub
# version
Version_id='2.11.1'
# value   [online | offline]
Version_line='offline'

# The domain name and CA info.
My_domain=$hostname1
TLS_C='CN'
TLS_ST='Beijing'
TLS_L='Beijing'
TLS_O='example'
TLS_OU='Personal'
TLS_CN='MyPersonal'

# HTTP  tag , 0 : not chang; 1 :  https ; 2 :http;   default 1
harbor_http=1
# online version tag
# download for github or local.
Online=false
osname=''
WithTrivy=true

file_name="harbor-${Version_line}-installer-v${Version_id}.tgz"

if  command uname -a | grep -q 'Debian' ; then
    osname='debian'
elif command uname -a | grep  -q 'Ubuntu' ; then
    osname='ubuntu'
fi

# check wgt/curl
if ! command -v wget >/dev/null 2>&1; then
    sudo apt-get install wget curl -y
fi

echo $file_name

# Download harbor
if  $Online ; then
   echo 'Downloading  harbor.'
   wget -c https://github.com/goharbor/harbor/releases/download/v${Version_id}/harbor-${Version_line}-installer-v${Version_id}.tgz
   echo 'Download harbor END. '
fi

if  [ -f "$file_name" ]; then
    echo "file:$file_name ok!"
else
    echo  "File not found."
    exit  0
fi
# tar harbor
tar xzvf $file_name
cd harbor
sudo cp harbor.yml.tmpl harbor.yml

# ENV Setting
if [ -d "/etc/modules" ]; then
   mkdir -p /etc/modules
   echo "kvm" | sudo tee -a /etc/modules
fi

if cat /proc/cpuinfo | grep -q 'Intel'; then
   echo "Intel CPU kvm Setting."
   echo "kvm_intel" | sudo tee -a /etc/modules
elif cat /proc/cpuinfo | grep -q 'AMD'; then
   echo "AMD CPU kvm setting."
   echo "kvm_amd" | sudo tee -a /etc/modules
fi

#  uninstall all conflicting packages:
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/$osname/gpg -o /etc/apt/keyrings/docker.asc
if [ -f "/etc/apt/keyrings/docker.asc" ]; then
        echo 'Docker asc file ok.'
else
        echo 'Restart download Docker asc file.'
        sudo wget -c  https://download.docker.com/linux/$osname/gpg -O /etc/apt/keyrings/docker.asc
fi
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$osname \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# Install Docker-eng
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo  ln -s /usr/libexec/docker/cli-plugins/docker-compose /usr/bin/docker-compose



if [ $harbor_http -eq 1 ]; then
   sudo openssl genrsa -out ca.key 4096

   sudo openssl req -x509 -new -nodes -sha512 -days 3650 \
    -subj "/C=${TLS_C}/ST=${TLS_ST}/L=${TLS_L}/O=${TLS_O}/OU=${TLS_OU}/CN=${TLS_CN} Root CA" \
    -key ca.key \
    -out ca.crt

   sudo openssl genrsa -out ${My_domain}.key 4096

   sudo openssl req -sha512 -new \
    -subj "/C=${TLS_C}/ST=${TLS_ST}/L=${TLS_L}/O=${TLS_O}/OU=${TLS_OU}/CN=${TLS_CN}" \
    -key ${My_domain}.key \
    -out ${My_domain}.csr


   sudo cat > v3.ext <<-EOF
        authorityKeyIdentifier=keyid,issuer
        basicConstraints=CA:FALSE
        keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
        extendedKeyUsage = serverAuth
        subjectAltName = @alt_names
        [alt_names]
          DNS.1= ${My_domain}
          DNS.2=localhost
          DNS.3=$hostname1
          IP.1=127.0.0.1
EOF

    sudo openssl x509 -req -sha512 -days 3650 \
    -extfile v3.ext \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -in ${My_domain}.csr \
    -out ${My_domain}.crt
    sudo mkdir -p /data/cert/
    sudo chown -R   ${username}:${groupname} /data/cert/
    sudo cp ${My_domain}.crt /data/cert/
    sudo cp ${My_domain}.key /data/cert/

    sudo openssl x509 -inform PEM -in ${My_domain}.crt -out ${My_domain}.cert
    sudo mkdir -p /etc/docker/certs.d/${My_domain}
    sudo cp ca.crt /etc/docker/cert.d/${My_domain}/
    sudo cp ${My_domain}.cert /etc/docker/certs.d/${My_domain}/
    sudo cp ${My_domain}.key /etc/docker/certs.d/${My_domain}/
    sudo cp ca.crt /etc/docker/certs.d/${My_domain}/
    sed -i "s/certificate: \/your\/certificate\/path/certificate: \/data\/cert\/${My_domain}.crt/" $pwddir/harbor/harbor.yml
    sed -i "s/private_key: \/your\/private\/key\/path/private_key: \/data\/cert\/${My_domain}.key/" $pwddir/harbor/harbor.yml

 
    sudo mkdir -p /etc/harbor/tls/internal
    sudo chown -R ${username}:${groupname} /etc/harbor/tls/internal
    sudo sudo openssl genrsa -out ca1.key 4096
    sudo openssl req -x509 -new -nodes -sha512 -days 3650 \
         -subj "/C=${TLS_C}/ST=${TLS_ST}/L=${TLS_L}/O=${TLS_O}/OU=${TLS_OU}/CN=${TLS_CN} Root CA" \
         -key ca1.key \
         -out ca1.crt
    sudo cp ca1.crt  /etc/harbor/tls/internal/ca.crt
    sudo cp ca1.key  /etc/harbor/tls/internal/ca.key
    sudo ./../build_ca.sh harbor_internal_ca  ${My_domain}
    sudo ./../build_ca.sh core
    sudo ./../build_ca.sh job_service jobservice
    sudo ./../build_ca.sh proxy
    sudo ./../build_ca.sh portal
    sudo ./..//build_ca.sh registry
    sudo ./../build_ca.sh registryctl
    sudo ./../build_ca.sh trivy_adapter 
    sudo cp  harbor_internal_ca.key /etc/harbor/tls/internal/harbor_internal_ca.key
    sudo cp  harbor_internal_ca.crt /etc/harbor/tls/internal/harbor_internal_ca.crt
    sudo cp  core.key /etc/harbor/tls/internal/core.key
    sudo cp  core.crt /etc/harbor/tls/internal/core.crt
    sudo cp  job_service.key /etc/harbor/tls/internal/job_service.key
    sudo cp  job_service.crt /etc/harbor/tls/internal/job_service.crt
    sudo cp  proxy.key /etc/harbor/tls/internal/proxy.key
    sudo cp  proxy.crt /etc/harbor/tls/internal/proxy.crt
    sudo cp  portal.key /etc/harbor/tls/internal/portal.key
    sudo cp  portal.crt /etc/harbor/tls/internal/portal.crt
    sudo cp  registry.key /etc/harbor/tls/internal/registry.key
    sudo cp  registry.crt /etc/harbor/tls/internal/registry.crt
    sudo cp  registryctl.key /etc/harbor/tls/internal/registryctl.key
    sudo cp  registryctl.crt /etc/harbor/tls/internal/registryctl.crt
    sudo cp  trivy_adapter.key /etc/harbor/tls/internal/trivy_adapter.key
    sudo cp  trivy_adapter.crt /etc/harbor/tls/internal/trivy_adapter.crt


    sudo mkdir -p /etc/harbor/tls/internal
    sudo chown -R ${username}:${groupname} /etc/harbor/tls/internal
    sed -i '/^#.*internal\_tls:$/s/^#/ /'  $pwddir/harbor/harbor.yml
    sed -i '35 s/#//' $pwddir/harbor/harbor.yml
    sed -i '/^#.*dir: \/etc\/harbor\/tls\/internal$/s/^#//'  $pwddir/harbor/harbor.yml
fi

if [ $harbor_http -eq 2 ]; then
    sed -i 's/https:/#https:/' $pwddir/harbor/harbor.yml
    sed -i 's/  port: 443/#&/g' $pwddir/harbor/harbor.yml
    sed -i 's/  certificate:/#&/g' $pwddir/harbor/harbor.yml
    sed -i 's/  private_key:/#&/g' $pwddir/harbor/harbor.yml
fi

# set domain
sed -i "s/hostname: reg.mydomain.com/hostname: ${My_domain}/" $pwddir/harbor/harbor.yml
if $WithTrivy ; then
# Harbor install --with-trivy
    sudo ./prepare.sh  --with-trivy
    sudo ./install.sh  --with-trivy
else
    sudo ./prepare.sh
    sudo ./install.sh
fi
# END
echo "Good Luck!"



