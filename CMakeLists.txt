cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pnjpeg)

find_package(Boost 1.65.1 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

if (NOT TARGET thread_pool)
    add_subdirectory(external/thread_pool EXCLUDE_FROM_ALL)
endif()
aux_source_directory(. DIR_SRCS)

add_executable(pnjpeg ${DIR_SRCS})
target_link_libraries(pnjpeg "${Boost_LIBRARIES}" thread_pool::thread_pool)
set_property(TARGET pnjpeg PROPERTY CXX_STANDARD 17)
