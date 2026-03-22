# CS6413
CS6413 Group Project

steps to compile:
1. go to build directory (cd build)
2. run cmake command wih the following flags: cmake -DWITH_PROCPS=OFF -DWITH_SUPERCOP=OFF -DOPENSSL_ROOT_DIR="/opt/homebrew/opt/openssl@3" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DUSE_ASM=OFF -DCURVE=ALT_BN128 ..
3. then run 'make' from inside build directory
4. if successful, the executable can be run as './src/main' (from within build directory)
