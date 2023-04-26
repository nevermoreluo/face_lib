cmake_minimum_required(VERSION 3.20)
project(Google_Tests_run)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../src)


file(GLOB_RECURSE Tests_Sources ./*.h ./*.cpp ./*.c)

# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Google_Tests_run ${Tests_Sources})

find_package(dlib CONFIG REQUIRED)
# set(OpenCV_DIR "C:/Users/SG220/source/repos/testCmakeProj/out/build/x86-Release/vcpkg_installed/x86-windows/share/opencv3")
find_package(OpenCV REQUIRED)
target_link_libraries(Google_Tests_run face gtest gtest_main dlib::dlib ${OpenCV_LIBS})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../res
  $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

if (MSVC)
    set_property(TARGET Google_Tests_run PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
endif()


enable_testing()

add_test(gtest ${PROJECT_NAME})

