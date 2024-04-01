cmake_minimum_required(VERSION 3.5)
project(NatNetws)

set(CMAKE_CXX_STANDARD 14)

set(EIGEN_INCLUDE_PATH "/usr/include/eigen3")


add_library(NatNetLibShared SHARED IMPORTED)
set_target_properties(NatNetLibShared PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/libNatNetLibShared.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/"
        )

INCLUDE_DIRECTORIES(
        include
        ${EIGEN_INCLUDE_PATH}
)

FILE(GLOB_RECURSE SOURCE_FILES src/*.c*)
ADD_EXECUTABLE(example
        samples/SampleClient/SampleClient.cpp)
TARGET_LINK_LIBRARIES(example
        NatNetLibShared)

ADD_EXECUTABLE(natnet_client
        ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(natnet_client
        NatNetLibShared
        pthread)

