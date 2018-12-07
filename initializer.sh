#! /bin/sh

# download clones, set up git
git config credential.helper store
git clone https://github.com/npfoss/Halite3RL.git
cd Halite3RL
git config credential.helper store
# git clone https://github.com/kentslaney/Halite3RL_sync sync
# cd sync
# git config user.name "RLaDOS"
# cd ..

sudo add-apt-repository universe
sudo apt-get update
sudo apt install python3-pip -y
sudo apt install unzip

# install script
pip3 install tensorflow==1.8 scipy zstd gym dill joblib
pip3 install gym[atari]

# find the OS
unameOut="$(uname -s)"
case "${unameOut}" in
Linux*)     machine=Linux;;
Darwin*)    machine=Mac;;
CYGWIN*)    machine=Cygwin;;
MINGW*)     machine=MinGw;;
*)          machine="UNKNOWN:${unameOut}"
esac
echo ${machine}

if [ "$machine" = "Mac" ]
then
    curl https://halite.io/assets/downloads/Halite3_Python3_MacOS.zip > Halite3_Python3.zip
else
    curl https://halite.io/assets/downloads/Halite3_Python3_Linux-AMD64.zip > Halite3_Python3.zip
fi

unzip Halite3_Python3.zip -d tmp_halite
mv tmp_halite/halite .
rm -rf tmp_halite/ Halite3_Python3.zip
