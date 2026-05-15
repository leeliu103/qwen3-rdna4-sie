cmake_minimum_required(VERSION 3.16)

project(llama_golden_tools LANGUAGES CXX)

set(LLAMA_CPP_DIR "/app/llama.cpp" CACHE PATH "Path to llama.cpp checkout")
set(LLAMA_CPP_BIN "${LLAMA_CPP_DIR}/build/bin" CACHE PATH "Path to llama.cpp built binaries and shared libraries")

add_executable(llama-golden-dump
    tools/llama_golden_dump.cpp
)

target_compile_features(llama-golden-dump PRIVATE cxx_std_17)
target_include_directories(llama-golden-dump PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
)
target_link_directories(llama-golden-dump PRIVATE
    "${LLAMA_CPP_BIN}"
)
target_link_libraries(llama-golden-dump PRIVATE
    llama
    ggml
    ggml-base
    ggml-cpu
)

if (EXISTS "${LLAMA_CPP_BIN}/libggml-hip.so")
    target_link_libraries(llama-golden-dump PRIVATE ggml-hip)
endif()

set_target_properties(llama-golden-dump PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_BIN}"
)
