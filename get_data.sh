mkdir -p data
pushd data
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar xvf images.tar
tar xvf annotation.tar
popd