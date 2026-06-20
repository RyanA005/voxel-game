cmake_minimum_required(VERSION 3.11)
project(voxel_parkour C)

set(CMAKE_C_STANDARD 99)

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

if(WIN32)
    target_link_libraries(voxel_parkour PRIVATE winmm)
endif()
