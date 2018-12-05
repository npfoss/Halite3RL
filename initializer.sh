#! /bin/sh

# download clones, set up git
git config credential.helper store
git clone https://github.com/npfoss/Halite3RL.git
cd Halite3RL
git clone https://github.com/kentslaney/Halite3RL_sync sync
cd sync
git config user.name "RLaDOS"
git remote set-url origin https://RLaDOS:538allspelledout@github.com/kentslaney/Halite3RL_sync #DO NOT STEAL PLZ
cd ..



# install script
pip3 install tensorflow scipy zstd gym dill gym[atari] joblib

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
    curl -O https://halite.io/assets/downloads/Halite3_Python3_MacOS.zip
    unzip Halite3_Python3_MacOS.zip -d tmp_halite
    mv tmp_halite/halite .
    rm -rf tmp_halite/ Halite3_Python3_MacOS.zip
else
    curl -O https://halite.io/assets/downloads/Halite3_Python3_Linux-AMD64.zip
    unzip Halite3_Python3_Linux-AMD64.zip -d tmp_halite
    mv tmp_halite/halite .
    rm -rf tmp_halite/ Halite3_Python3_Linux-AMD64.zip
fi

