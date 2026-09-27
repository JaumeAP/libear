include(${PROJECT_SOURCE_DIR}/submodules/catch2.cmake)

# --- unit tests ---
function(add_ear_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name}
    PRIVATE
    ear
    catch2
    Eigen3::Eigen
  )
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/submodules
    ${PROJECT_SOURCE_DIR}/src
  )
  add_test(
    NAME ${name}
    COMMAND ${name}
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/src"
  )
endfunction()

# copy test files so unit test can find them relative to their running location
# when executed as "test" target
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/test_data" DESTINATION ${PROJECT_BINARY_DIR})

if(EAR_INTERNAL_TESTS)
    add_ear_test("decorrelate_tests")
    add_ear_test("extent_tests")
    target_sources(extent_tests PRIVATE "reference/extent.cpp")
    add_ear_test("gain_calculator_direct_speakers_tests")
    add_ear_test("geom_tests")
    add_ear_test("hoa_tests")
    add_ear_test("point_source_panner_tests")
endif()

add_ear_test("block_convolver_tests")
target_sources("block_convolver_tests" PRIVATE "block_convolver_test_utils.cpp")

if(EAR_USE_VDSP)
  add_ear_test("fft_vdsp_tests")

  # same block convolver tests, run with the vDSP FFT implementation
  add_executable(block_convolver_vdsp_tests block_convolver_tests.cpp block_convolver_test_utils.cpp)
  target_link_libraries(block_convolver_vdsp_tests PRIVATE ear catch2 Eigen3::Eigen)
  target_include_directories(block_convolver_vdsp_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/submodules
    ${PROJECT_SOURCE_DIR}/src
  )
  target_compile_definitions(block_convolver_vdsp_tests PRIVATE BLOCK_CONVOLVER_TEST_VDSP)
  add_test(
    NAME block_convolver_vdsp_tests
    COMMAND $<TARGET_FILE:block_convolver_vdsp_tests>
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/src"
  )
endif()
add_ear_test("bs2051_tests")
add_ear_test("conversion_tests")
add_ear_test("delay_buffer_tests")
add_ear_test("eigen_helpers_tests")
add_ear_test("gain_calculator_hoa_tests")
add_ear_test("gain_calculator_objects_tests")
add_ear_test("gain_interpolator_tests")
add_ear_test("variable_block_size_tests")
