cmake_minimum_required(VERSION 3.10)
project(
    test
    VERSION 0.0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

function(normalize_string INPUT OUTPUT)
    string(TOLOWER ${INPUT} _value)
    string(REGEX REPLACE "[^a-z0-9_]+" "_" _value ${_value})
    string(REGEX REPLACE "_+" "_" _value ${_value})
    string(REGEX REPLACE "^_|_$" "" _value ${_value})
    set(${OUTPUT} ${_value} PARENT_SCOPE)
endfunction()

normalize_string(${PROJECT_NAME} IDENTITY)

find_package(OpenCV REQUIRED)

if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH "x86_64-linux")
elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH "aarch64-linux")
else()
    message(FATAL_ERROR, "Unknown host architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

set(ACL_HOME $ENV{ASCEND_TOOLKIT_HOME})

file(GLOB LIB_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
message(STATUS ${LIB_FILES})
add_library(lib SHARED)
target_sources(lib PRIVATE ${LIB_FILES})
target_compile_features(lib PUBLIC cxx_std_17)
target_include_directories(lib
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${ACL_HOME}/${ARCH}/include"
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lib
    PUBLIC
        "${ACL_HOME}/${ARCH}/lib64/libascendcl.so"
        "${ACL_HOME}/${ARCH}/lib64/libacl_dvpp.so"
        "${ACL_HOME}/${ARCH}/lib64/libopapi_cv.so"
        ${OpenCV_LIBRARIES}
)
set_target_properties(lib
    PROPERTIES
        OUTPUT_NAME ${IDENTITY}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
)

add_executable(demo ${CMAKE_SOURCE_DIR}/main.cpp)
target_link_libraries(demo
    PRIVATE
        lib   
)
set_target_properties(demo
    PROPERTIES
        OUTPUT_NAME ${IDENTITY}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

message(STATUS "Shared library target : ${IDENTITY}")
message(STATUS "Shared library output : ${CMAKE_SOURCE_DIR}/lib")
message(STATUS "Demo output           : ${CMAKE_SOURCE_DIR}/bin")