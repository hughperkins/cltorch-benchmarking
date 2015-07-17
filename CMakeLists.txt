cmake_minimum_required(VERSION 2.8)

set(HOME $ENV{HOME})
message("HOME ${HOME}")
find_path(EASYCL_INCLUDE
    NAMES EasyCL.h
    PATHS
        /usr/include/easycl
        /usr/local/include/easycl
        /opt/EasyCL/include/easycl
        ${HOME}/git/EasyCL/dist/include/easycl
    DOC "OpenCL header file path"
)
find_path(CLEW_INCLUDE
    NAMES clew.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/EasyCL/include
        ${HOME}/git/EasyCL/dist/include
    DOC "clew header file path"
)
find_library(
  EasyCL
  NAME EasyCL
  PATHS ${HOME}/git/EasyCL/dist/lib
  /opt/EasyCL/lib
  /usr/lib
  /usr/local/lib
)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message("EasyCL ${EasyCL}")
include_directories(${EASYCL_INCLUDE} ${CLEW_INCLUDE})
link_libraries(${EasyCL})

add_executable(test_launch test_launch.cpp)
add_executable(test_apply1 test_apply1.cpp)
add_executable(test_apply1b test_apply1b.cpp)
add_executable(test_applystrided test_applystrided.cpp)

