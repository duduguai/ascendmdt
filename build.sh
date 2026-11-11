ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
cd $ROOT
if [ ! -d "build" ]; then
    mkdir build
fi
cd build

cmake .. && make
cd .. && rm -r build