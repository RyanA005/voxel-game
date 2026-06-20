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
    neural_simd.c
    sim.c
)

add_executable(voxel_parkour ${SOURCES})
target_include_directories(voxel_parkour PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(voxel_parkour PRIVATE raylib)

if(MSVC)
    target_compile_options(voxel_parkour PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2 /fp:fast>)
else()
    target_compile_options(voxel_parkour PRIVATE
        $<$<NOT:$<CONFIG:Debug>>:-O3 -ffast-math -funroll-loops>)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            target_compile_options(voxel_parkour PRIVATE $<$<NOT:$<CONFIG:Debug>>:-march=native>)
        endif()
        check_c_compiler_flag("-mavxvnni" COMPILER_SUPPORTS_AVXVNNI)
        if(COMPILER_SUPPORTS_AVXVNNI)
            set_source_files_properties(neural_simd.c PROPERTIES
                COMPILE_FLAGS "-mavx2 -mfma -mavxvnni")
        else()
            set_source_files_properties(neural_simd.c PROPERTIES
                COMPILE_FLAGS "-mavx2 -mfma")
        endif()
    endif()
endif()

if(WIN32)
    target_link_libraries(voxel_parkour PRIVATE winmm)
endif()
