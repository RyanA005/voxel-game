cmake_minimum_required(VERSION 3.11)
project(voxel_parkour C)

set(CMAKE_C_STANDARD 99)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
)
FetchContent_MakeAvailable(raylib)

set(SOURCES
    main.c
    world.c
    physics.c
    observation.c
    neural.c
    sim.c
)

add_executable(voxel_parkour ${SOURCES})
target_include_directories(voxel_parkour PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(voxel_parkour PRIVATE raylib)

if(MSVC)
    target_compile_options(voxel_parkour PRIVATE /O2 /fp:fast)
else()
    target_compile_options(voxel_parkour PRIVATE -O3 -ffast-math -fno-math-errno)
    set_source_files_properties(neural.c PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

if(WIN32)
    target_link_libraries(voxel_parkour PRIVATE winmm)
endif()
