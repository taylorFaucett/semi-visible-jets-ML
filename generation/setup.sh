# base for most cvmfs packages
ABASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/x86_64

# gcc 4.9
GCC_BASE=${ABASE}/Gcc/gcc493_x86_64_slc6/slc6/gcc49
GCC_BIN=${GCC_BASE}/bin
export PATH=${GCC_BIN}:$PATH
export CC=$GCC_BIN/gcc
export CPP=$GCC_BIN/cpp
export CXX=$GCC_BIN/g++
alias gcc=$CC
alias g++=$CXX
export LD_LIBRARY_PATH=${GCC_BASE}/lib64:${GCC_BASE}/lib:${LD_LIBRARY_PATH}

# Boost
BOOST_BASE=boost-1.55.0-python2.7-x86_64-slc6-gcc48
export BOOST_PATH=$ABASE/boost/$BOOST_BASE/$BOOST_BASE
export LD_LIBRARY_PATH=$BOOST_PATH/lib:${LD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=$BOOST_PATH/include:${CPLUS_INCLUDE_PATH}

## root setup
export ROOTSYS=${ABASE}/root/6.04.14-x86_64-slc6-gcc49-opt
source $ROOTSYS/bin/thisroot.sh
#cd $ROOTSYS
#. bin/thisroot.sh
#cd ~/

# # export PATH=~/.local/bin${PATH:+:$PATH}
