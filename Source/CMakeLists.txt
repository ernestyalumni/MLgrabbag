CMAKE_MINIMUM_REQUIRED (VERSION 3.16)

# Handle CMAKE_CUDA_ARCHITECTURES
# See https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html

MESSAGE("CMAKE_CUDA_ARCHITECTURES before SET: " ${CMAKE_CUDA_ARCHITECTURES})

IF (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Maxwell cards (CUDA 6 until CUDA 11)
  # SM52 or SM_52, compute_52
  # GeForce 900, GTX-970, GTX-980, GeForce GTX 980 Ti
  #
  # Pascal (CUDA 8 and later)
  # SM61 or SM_61, compute_61
  # GTX 1080, GTX 1070, GTX 1060, GTX 1050
  # SM62 or SM_62, compute_62
  # Tegra X2, Jetson TX2
  #
  # Ampere (CUDA 11.1 and later)
  # SM86 or SM_86, compute_86
  # RTX 3080, RTX 3070, RTX 3050
  SET(CMAKE_CUDA_ARCHITECTURES 52 60 61 62 75 86)
ELSE()
  SET(CMAKE_CUDA_ARCHITECTURES 52 60 61 62 75 86)
ENDIF()

MESSAGE("CMAKE_CUDA_ARCHITECTURES after SET: " ${CMAKE_CUDA_ARCHITECTURES})

# Language of the project is CXX (for C++) and CUDA (C++)
PROJECT(STUNTICONS LANGUAGES CXX CUDA)

# BEFORE prepends the directories specified onto current list of directories.
# cf. https://cmake.org/cmake/help/v3.0/command/include_directories.html
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR})

# Include will load and run CMake code of this module, "CheckCXXCompilerFlag."
# Without this, you'll get CMake Error, Unknown CMake command
# "CHECK_CXX_COMPILER_FLAG"
INCLUDE(CheckCXXCompilerFlag)

################################################################################
## Compiler Flags
################################################################################

CHECK_CXX_COMPILER_FLAG(-std=c++20 COMPILER_SUPPORTS_CXX20)

# Check if Compiler supports the Working draft for ISO C++ 2020 standard.
CHECK_CXX_COMPILER_FLAG(-std=c++2a COMPILER_SUPPORTS_CXX2A)

IF(COMPILER_SUPPORTS_CXX20)
  SET(CMAKE_CXX_STANDARD "20") # C++20
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
ELSEIF(COMPILER_SUPPORTS_CXX2A)
  SET(CMAKE_CXX_STANDARD "20") # C++20, but with c++2a
  MESSAGE("Setting to the Working draft of C++20: " ${COMPILER_SUPPORTS_CXX2A})
  SET(CMAKE_CXX_STANDARD_REQUIRED ON) # ... is required
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
ELSE()
  SET(CMAKE_CXX_STANDARD "17")
  SET(CMAKE_CXX_STANDARD_REQUIRED ON)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

MESSAGE("CMAKE_CXX_STANDARD: " ${CMAKE_CXX_STANDARD})

MESSAGE("We will now look for Cuda; this is the CMAKE_MODULE_PATH: "
  ${CMAKE_MODULE_PATH})
MESSAGE("We will now look for Cuda; this is the CMAKE_PREFIX_PATH: "
  ${CMAKE_PREFIX_PATH})

################################################################################
## CUDA package
################################################################################

IF (CUDA_FOUND)
  MESSAGE(STATUS "Cuda was found.")
ENDIF()

# See https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
# FindCUDAToolkit new in 3.17

IF (NOT (CMAKE_VERSION VERSION_LESS "3.17.0"))

  FIND_PACKAGE(CUDAToolkit)

  IF (CUDAToolkit_FOUND)
    MESSAGE(STATUS "Cuda Toolkit was found at ${CUDA_TOOLKIT_ROOT_DIR}.")
  ELSE()
    MESSAGE(STATUS "Cuda Toolkit was not found.")
  ENDIF()

  MESSAGE("CUDAToolkit_VERSION: " ${CUDAToolkit_VERSION})

ENDIF()

MESSAGE("CMAKE_CUDA_STANDARD before SET: " ${CMAKE_CUDA_STANDARD})

# See https://gitlab.kitware.com/cmake/cmake/-/issues/23079

IF (NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  # Since the CUDA20 variables exist the paired language behavior in PROJECT(
  # ... LANGUAGES CXX CUDA) will think compiler supports CUDA20 but won't add
  # anything to compile line.
  MESSAGE("CMAKE_CUDA20_STANDARD_COMPILE_OPTION was not defined")
  SET(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  SET(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
ENDIF()

MESSAGE("CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})

IF (NOT (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0) AND
  COMPILER_SUPPORTS_CXX20)
  MESSAGE("Set CUDA for C++20")
  SET(CMAKE_CUDA_STANDARD 20)
  SET(CMAKE_CUDA_FLAGS "-std=c++20")
ENDIF()

MESSAGE("CMAKE_CUDA_STANDARD after SET: " ${CMAKE_CUDA_STANDARD})

################################################################################
## CUDNN package
################################################################################

# You make have to change this manually for your setup of where CUDNN's
# libcudnn.so was installed
# https://forums.developer.nvidia.com/t/cudnn-linking-error-using-cmake/57968
INCLUDE_DIRECTORIES(/usr/lib/x86_64-linux-gnu)

################################################################################
## CCACHE package
################################################################################

FIND_PROGRAM(CCACHE_FOUND ccache)

IF(CCACHE_FOUND)
  SET(CMAKE_CXX_COMPILER_LAUNCHER ccache)
ENDIF()

# Declare dependency on GoogleTest which is downloaded from GitHub.
# cf. https://google.github.io/googletest/quickstart-cmake.html
INCLUDE(FetchContent)

MESSAGE("CMAKE_VERSION: " ${CMAKE_VERSION})
IF (CMAKE_VERSION VERSION_LESS "3.23.0")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
ELSE()
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    # See https://github.com/aminya/project_options/issues/146
    # See https://cmake.org/cmake/help/latest/policy/CMP0135.html
    DOWNLOAD_EXTRACT_TIMESTAMP true
  )
ENDIF()

# If you have problems running `cmake ../Source` for instance, it's indicative
# of a stale CMake configuration, so try a hard removal of the build folder.

FetchContent_MakeAvailable(googletest)
