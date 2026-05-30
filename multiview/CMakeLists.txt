# ---------------------------------------------------------------------------
# multiview — multi-camera capture & fusion (see MULTIVIEW_PLAN.md)
#
# So far: calib — a self-contained Stereolabs/Matlab `.calib` reader/writer
# (intrinsics + distortion + extrinsics).  No external dependencies.
# ---------------------------------------------------------------------------

add_library(multiview_calib STATIC calib.c)

target_include_directories(multiview_calib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(multiview_calib PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

# Unit test: parse the committed fixture and check the values.
add_executable(calib_test calib_test.c)
target_link_libraries(calib_test PRIVATE multiview_calib m)
add_test(NAME calib_test
         COMMAND calib_test ${CMAKE_CURRENT_SOURCE_DIR}/test_data/sample.calib)

# ---------------------------------------------------------------------------
# aruco_qr — ArUco pose + QR timecode detection (needs the aruco/objdetect
# OpenCV modules; requested here so the core targets keep their own component
# set unchanged).
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d aruco objdetect videoio)

add_library(multiview_aruco_qr STATIC aruco_qr.cpp)
target_include_directories(multiview_aruco_qr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(multiview_aruco_qr PUBLIC multiview_calib ${OpenCV_LIBS})
target_compile_options(multiview_aruco_qr PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
)

add_executable(aruco_qr_probe aruco_qr_probe.cpp)
target_link_libraries(aruco_qr_probe PRIVATE multiview_aruco_qr ${OpenCV_LIBS})

message(STATUS "  multiview: multiview_calib + multiview_aruco_qr (libs), calib_test, aruco_qr_probe")
