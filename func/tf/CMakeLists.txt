set(FAASM_USER tf)

set(TF_LIB ${CMAKE_SYSROOT}/lib/wasm32-wasi/libtensorflow-lite.a)

set(TF_DIR /code/examples/examples/tensorflow)
set(TF_LITE_DIR ${TF_DIR}/tensorflow/lite)
set(TF_LITE_DOWNLOAD_DIR ${TF_LITE_DIR}/tools/make/downloads)
set(TF_BUILD_DIR ${TF_LITE_DIR}/tools/make/gen/)


include_directories(
        ${TF_DIR}
        ${TF_LITE_DIR}
        ${TF_LITE_DOWNLOAD_DIR}
        ${TF_LITE_DOWNLOAD_DIR}/eigen
        ${TF_LITE_DOWNLOAD_DIR}/absl
        ${TF_LITE_DOWNLOAD_DIR}/gemmlowp
        ${TF_LITE_DOWNLOAD_DIR}/neon_2_sse
        ${TF_LITE_DOWNLOAD_DIR}/farmhash/src
        ${TF_LITE_DOWNLOAD_DIR}/flatbuffers/include
)


add_library(tf_img_lib
        bitmap_helpers.cpp
        get_top_n.h
        get_top_n_impl.h
        image.h
        )


faasm_example_func(check check.cpp)
faasm_example_func(image_state image_state.cpp)
faasm_example_func(image_zygote image_zygote.cpp)
faasm_example_func(image_nostate image_nostate.cpp)

set(TF_HEADERS
    ${CMAKE_SYSROOT}/include/tensorflow
    ${CMAKE_SYSROOT}/include/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
)

target_include_directories(tf_check PUBLIC ${TF_HEADERS})
target_link_libraries(tf_check ${TF_LIB})

target_include_directories(tf_image_state PUBLIC ${TF_HEADERS}
)
target_link_libraries(tf_image_state ${TF_LIB} tf_img_lib)

target_include_directories(tf_image_zygote PUBLIC ${TF_HEADERS}
)
target_link_libraries(tf_image_zygote ${TF_LIB} tf_img_lib)


target_include_directories(tf_image_nostate PUBLIC ${TF_HEADERS}
)
target_link_libraries(tf_image_nostate ${TF_LIB} tf_img_lib)
