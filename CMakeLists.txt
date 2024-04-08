cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(matching VERSION 1.0.0)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
	set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdParty/opencv)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 17)

find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

option(BUILD_SHARED_LIBS "Specifies the type of libraries (SHARED or STATIC) to build" ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/3rdParty/spdlog/include
	include
	matcher
	)

add_executable(demo "demo.cpp")
target_link_libraries(demo ${OpenCV_LIBS})
if(UNIX)
	target_link_libraries(demo dl)
endif()

#安装规则
install(TARGETS demo 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES 
    "demo.cpp"
	DESTINATION ${CMAKE_INSTALL_PREFIX}
)

add_subdirectory(matcher)
