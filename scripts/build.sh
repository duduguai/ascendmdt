ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && cd .. && pwd)
cd $ROOT
if [ ! -d "build" ]; then
    mkdir build
fi
cd build

cmake .. && make