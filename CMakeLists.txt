cmake_minimum_required(VERSION 3.10)
project(
    Ascend-Model-Deploy-ToolKit
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

# file(GLOB LIB_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
set(LIB_FILES ${CMAKE_SOURCE_DIR}/src/aclsession/aclsession_core.cpp ${CMAKE_SOURCE_DIR}/src/aclsession/aclsession_yolo.cpp ${CMAKE_SOURCE_DIR}/src/modelcore/models_core_yolo.cpp ${CMAKE_SOURCE_DIR}/src/models/models_yolo.cpp)
message(STATUS ${LIB_FILES})
add_library(lib0 SHARED)
target_sources(lib0 PRIVATE ${LIB_FILES})
target_compile_features(lib PUBLIC cxx_std_17)
target_include_directories(lib0
    PUBLIC
        "${CMAKE_SOURCE_DIR}/inc/aclsession/core"
        "${CMAKE_SOURCE_DIR}/inc/aclsession/yolo"
        "${CMAKE_SOURCE_DIR}/inc/modelcore/yolo"
        "${CMAKE_SOURCE_DIR}/inc/models/yolo"
        "${CMAKE_SOURCE_DIR}/inc/tools"
        "${ACL_HOME}/${ARCH}/include"
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(lib0
    PUBLIC
        "${ACL_HOME}/${ARCH}/lib64/libascendcl.so"
        "${ACL_HOME}/${ARCH}/lib64/libacl_dvpp.so"
        "${ACL_HOME}/${ARCH}/lib64/libopapi_cv.so"
        ${OpenCV_LIBRARIES}
)
set_target_properties(lib0
    PROPERTIES
        OUTPUT_NAME "_yolo"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/${ARCH}"
)

message(STATUS "Shared library output : ${CMAKE_SOURCE_DIR}/lib/${ARCH}")