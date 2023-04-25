# CMakeList.txt : CMake project for testCmakeProj, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)
project(face)

add_library(${PROJECT_NAME} STATIC face.cpp face.h)

find_package(dlib CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE dlib::dlib)

# set(OpenCV_DIR "C:/Users/SG220/source/repos/testCmakeProj/out/build/x86-Release/vcpkg_installed/x86-windows/share/opencv3")
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
# TODO: Add tests and install targets if needed.

if (MSVC)
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()