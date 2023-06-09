# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.8)

set(swintransformer_trt_files
  swinTransformerPlugin.cpp
  swinTransformerINT8Plugin.cpp
)


if(BUILD_TRT)
  set(LIB_NAME "swinTransformer_plugin")
  add_library(${LIB_NAME} SHARED ${swintransformer_trt_files})
  set_target_properties(${LIB_NAME} PROPERTIES
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${LIB_NAME} trt_fused_multi_head_attention Swin SwinINT8 -lcudnn -lcublas -lcudart -lnvinfer)
endif()
