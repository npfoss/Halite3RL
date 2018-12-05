rm -f halite.zip
zip -r halite.zip . -x@exclude.lst

if [ ! -d dependencies ]; then
  mkdir dependencies;
fi
if [ ! -d dependencies/baselines/acer ]; then
  mkdir -p dependencies/baselines/acer;
fi
cd dependencies

pip3 install --no-dependencies --target . dill gym joblib
pip3 install --no-dependencies --target baselines/acer dill

zip -r ../halite.zip *
