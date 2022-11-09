project(myproject LANGUAGES CXX)

cmake_minimum_required(VERSION 3.8.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(UNIX_COMMON_COMPILE_OPTIONS -Wall -Wextra -pedantic -Wcast-qual
                                -Wdisabled-optimization -Wformat -Winit-self -Woverloaded-virtual
                                -Wredundant-decls -Wshadow -Wundef -Wno-unused -Wno-variadic-macros
                                -Wfloat-equal -Werror -Wold-style-cast)
set(UNIX_HOST_COMPILE_OPTIONS ${UNIX_COMMON_COMPILE_OPTIONS}
                              -Wcast-align -Wctor-dtor-privacy -Wmissing-include-dirs
                              -Wsign-promo -Wno-error=missing-include-dirs)
set(UNIX_DEVICE_COMPILE_OPTIONS ${UNIX_COMMON_COMPILE_OPTIONS}
                                -Wno-error=sign-promo -Wno-error=delete-non-virtual-dtor)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    set(UNIX_COMMON_COMPILE_OPTIONS "${UNIX_COMMON_COMPILE_OPTIONS} -Wno-deprecated-copy")
endif()

set(COMPILE_OPTIONS "")

set(ARCH "IMX8" CACHE STRING "default value of processor architecture")
if(${ARCH} MATCHES "PC")
    set(COMPILE_OPTIONS ${UNIX_HOST_COMPILE_OPTIONS})
else()
    set(COMPILE_OPTIONS ${UNIX_DEVICE_COMPILE_OPTIONS})
endif()

# Turn on automatic invocation of the MOC, UIC & RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if ("${CMAKE_INSTALL_DATAROOTDIR}" STREQUAL "")
	set(APP_DATA_DIR ..)
	set(APP_DATA_DIR_DEF ..)
else()
	set(APP_DATA_DIR ${CMAKE_INSTALL_DATAROOTDIR}/stereo-vision)
	set(APP_DATA_DIR_DEF ${CMAKE_INSTALL_PREFIX}/${APP_DATA_DIR})
endif()
add_compile_definitions(APP_DATA_DIR="${APP_DATA_DIR_DEF}")

include_directories(${CMAKE_SOURCE_DIR})

find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)
find_package(OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(StereoVision 
    main.cpp
    mainwindow.cpp
    mainwindow.ui
    facedetection.cpp
    auxutils.cpp
    tensorflow.cpp
    calibration.cpp
    takephoto.cpp
)

# set tensorflow lib and include directory
if(${ARCH} MATCHES "PC")
    add_compile_definitions(ARCH_PC)
    set(TENSORFLOW_DIR "/home/mpiccard/Projects/AventEdgeAI/tensorflow" CACHE STRING "directory of tensorflow lite")
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR})
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/eigen )
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/absl)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/gemmlowp)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/neon_2_sse)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/farmhash/src)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
    target_include_directories(StereoVision PUBLIC ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib)
    set(TFLITE_LIB ${TENSORFLOW_DIR}/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a)
else()
    set(TFLITE_LIB "libtensorflow-lite.so")
endif()

target_link_libraries(StereoVision
    Qt5::Core
    Qt5::Widgets
    Qt5::OpenGL
    ${OpenCV_LIBS}
    ${TFLITE_LIB}
    pthread
)

install(TARGETS StereoVision 
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY dnnData
    DESTINATION ${APP_DATA_DIR}
    )

