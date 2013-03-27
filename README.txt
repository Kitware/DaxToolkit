
################################################################################
##                             Build Dax Toolkit                              ##
################################################################################
git clone git://github.com/Kitware/DaxToolkit.git dax
mkdir dax-build
cd dax-build
cmake-gui ../dax
make
ctest

################################################################################
##                                Dependencies                                ##
################################################################################

1. CMake 2.8.8 (http://cmake.org/cmake/resources/software.html)
   To build the Dax testing framework you need a relatively new version of CMake
   We recommend 2.8.10 but support back to 2.8.8
2. Boost 1.49.0 or greater (http://www.boost.org)
   We only require that you install the header components of Boost
3. Cuda Toolkit 4+ or Thrust 1.4 / 1.5
   (https://developer.nvidia.com/cuda-toolkit)
   (https://thrust.github.com)
   For the CUDA backend you will need at least the CudaToolkit 4 and the
   corresponding device driver. If you don't have a NVidia graphics card and
   want to use OpenMP acceleration, you will need Thrust version 1.4 or 1.5.
   We currently haven't tested with Thrust 1.6

################################################################################
##                              Supported OSes                                ##
################################################################################

Currently Supported:
1. Linux
2. OSX: Snow leopard and up

We are currently working on Windows support, and hope to have it finished soon.

################################################################################
##                                 About Dax                                  ##
################################################################################
While the Dax toolkit is header only, the repository includes a large testing
framework that needs to be built. Building Dax will verify that you have
enabled the correct options, and is a great resource for how to develop
your own Dax worklets.

Always configure Dax so that the build tree is not in the source directory.
The project will not run if you setup the source and build directories
to be the same directory.

################################################################################
##                                Documentation                               ##
################################################################################

For more information about how to write dax worklets and using dax from inside
your project we recommend you read the dax website at:

daxtoolkit.github.com
