
```
 2008  git clone git@github.com:ernestyalumni/openxla-nvgpu.git

 2016  mkdir BuildDevelopment
 2017  cd BuildDevelopment/

 2031  sudo apt-get install libstdc++-12-dev
 2032  cmake -GNinja -B BuildDevelopment/ -S .     -DCMAKE_BUILD_TYPE=RelWithDebInfo     -DIREE_ENABLE_ASSERTIONS=ON     -DIREE_ENABLE_LLD=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache


 2039  sudo apt-get install lld
```