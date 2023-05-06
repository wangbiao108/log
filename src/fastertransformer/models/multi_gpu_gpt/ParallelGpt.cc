/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <unistd.h>
#include <sys/syscall.h>

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gen_relative_pos_bias.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/logprob_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

namespace fastertransformer {

template<typename T>
void ParallelGpt<T>::initialize()
{
    printf("*******************************************************************************************\n");
    printf("ParallelGptContextDecoder参数:\n");
    printf("[%s:%d:pid%d]: max_batch_size               :%d\n",  __FUNCTION__,  __LINE__, getpid(), 0                            );
    printf("[%s:%d:pid%d]: max_seq_len                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), 0                            );
    printf("[%s:%d:pid%d]: head_num_                    :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)head_num_           );
    printf("[%s:%d:pid%d]: size_per_head_               :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)size_per_head_      );
    printf("[%s:%d:pid%d]: inter_size_                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)inter_size_         );
    printf("[%s:%d:pid%d]: num_layer_                   :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)num_layer_          );
    printf("[%s:%d:pid%d]: expert_num_                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)expert_num_         );
    printf("[%s:%d:pid%d]: moe_k_                       :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)moe_k_              );
    printf("[%s:%d:pid%d]: moe_layer_index_             :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)moe_layer_index_.size());
    printf("[%s:%d:pid%d]: layernorm_eps_               :%f\n",  __FUNCTION__,  __LINE__, getpid(), layernorm_eps_               );

    printf("[%s:%d:pid%d]: gpt_variant_params_:\n",  __FUNCTION__,  __LINE__, getpid());
    printf("[%s:%d:pid%d]: layernorm_eps                :%f\n",  __FUNCTION__,  __LINE__, getpid(), gpt_variant_params_.layernorm_eps);
    printf("[%s:%d:pid%d]: layernorm_type               :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.layernorm_type);
    printf("[%s:%d:pid%d]: activation_type              :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.activation_type);
    printf("[%s:%d:pid%d]: has_positional_encoding      :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.has_positional_encoding);
    printf("[%s:%d:pid%d]: has_pre_decoder_layernorm    :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.has_pre_decoder_layernorm);
    printf("[%s:%d:pid%d]: has_post_decoder_layernorm   :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.has_post_decoder_layernorm);
    printf("[%s:%d:pid%d]: has_adapters                 :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.has_adapters);
    printf("[%s:%d:pid%d]: adapter_inter_size           :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.adapter_inter_size);
    printf("[%s:%d:pid%d]: use_attention_linear_bias    :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.use_attention_linear_bias);
    printf("[%s:%d:pid%d]: \n",  __FUNCTION__,  __LINE__, getpid());
   
    printf("[%s:%d:pid%d]: tensor_para_.rank_           :%d\n",  __FUNCTION__,  __LINE__, getpid(), tensor_para_.rank_      );
    printf("[%s:%d:pid%d]: tensor_para_.world_size_     :%d\n",  __FUNCTION__,  __LINE__, getpid(), tensor_para_.world_size_);
    printf("[%s:%d:pid%d]: pipeline_para_.rank_         :%d\n",  __FUNCTION__,  __LINE__, getpid(), pipeline_para_.rank_      );
    printf("[%s:%d:pid%d]: pipeline_para_.world_size_   :%d\n",  __FUNCTION__,  __LINE__, getpid(), pipeline_para_.world_size_);

    printf("[%s:%d:pid%d]: stream_                      :%p\n",  __FUNCTION__,  __LINE__, getpid(), stream_                      );
    printf("[%s:%d:pid%d]: cublas_wrapper_              :%p\n",  __FUNCTION__,  __LINE__, getpid(), cublas_wrapper_              );
    printf("[%s:%d:pid%d]: allocator_                   :%p\n",  __FUNCTION__,  __LINE__, getpid(), allocator_                   );
    printf("[%s:%d:pid%d]: is_free_buffer_after_forward_:%d\n",  __FUNCTION__,  __LINE__, getpid(), is_free_buffer_after_forward_);
    printf("[%s:%d:pid%d]: is_context_qk_buf_float_     :%d\n",  __FUNCTION__,  __LINE__, getpid(), is_context_qk_buf_float_     );
    printf("[%s:%d:pid%d]: attention_type_              :%d\n",  __FUNCTION__,  __LINE__, getpid(), attention_type_              );
    printf("[%s:%d:pid%d]: sparse_                      :%d\n",  __FUNCTION__,  __LINE__, getpid(), sparse_                      );
    printf("[%s:%d:pid%d]: int8_mode_                   :%d\n",  __FUNCTION__,  __LINE__, getpid(), int8_mode_                   );
    printf("[%s:%d:pid%d]: custom_all_reduce_comm_      :%p\n",  __FUNCTION__,  __LINE__, getpid(), custom_all_reduce_comm_      );
    printf("[%s:%d:pid%d]: enable_custom_all_reduce_    :%d\n",  __FUNCTION__,  __LINE__, getpid(), enable_custom_all_reduce_    );
    printf("*******************************************************************************************\n");
    gpt_context_decoder_ = new ParallelGptContextDecoder<T>(0,
                                                            0,
                                                            head_num_,
                                                            size_per_head_,
                                                            inter_size_,
                                                            num_layer_,
                                                            expert_num_,
                                                            moe_k_,
                                                            moe_layer_index_,
                                                            layernorm_eps_,
                                                            gpt_variant_params_,
                                                            tensor_para_,
                                                            pipeline_para_,
                                                            stream_,
                                                            cublas_wrapper_,
                                                            allocator_,
                                                            is_free_buffer_after_forward_,
                                                            is_context_qk_buf_float_,
                                                            attention_type_,
                                                            sparse_,
                                                            int8_mode_,
                                                            custom_all_reduce_comm_,
                                                            enable_custom_all_reduce_);

    printf("*******************************************************************************************\n");
    printf("ParallelGptDecoder参数:\n");
    printf("[%s:%d:pid%d]: max_batch_size               :%d\n",  __FUNCTION__,  __LINE__, getpid(), 0                            );
    printf("[%s:%d:pid%d]: head_num_                    :%d\n",  __FUNCTION__,  __LINE__, getpid(), head_num_                    );
    printf("[%s:%d:pid%d]: size_per_head_               :%d\n",  __FUNCTION__,  __LINE__, getpid(), size_per_head_               );
    printf("[%s:%d:pid%d]: inter_size_                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), inter_size_                  );
    printf("[%s:%d:pid%d]: num_layer_                   :%d\n",  __FUNCTION__,  __LINE__, getpid(), num_layer_                   );
    printf("[%s:%d:pid%d]: expert_num_                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), expert_num_                  );
    printf("[%s:%d:pid%d]: moe_k_                       :%d\n",  __FUNCTION__,  __LINE__, getpid(), moe_k_                       );
    printf("[%s:%d:pid%d]: moe_layer_index_             :%d\n",  __FUNCTION__,  __LINE__, getpid(), moe_layer_index_             );
    printf("[%s:%d:pid%d]: layernorm_eps_               :%d\n",  __FUNCTION__,  __LINE__, getpid(), layernorm_eps_               );
    printf("[%s:%d:pid%d]: gpt_variant_params_          :%d\n",  __FUNCTION__,  __LINE__, getpid(), gpt_variant_params_          );
    printf("[%s:%d:pid%d]: tensor_para_                 :%d\n",  __FUNCTION__,  __LINE__, getpid(), tensor_para_                 );
    printf("[%s:%d:pid%d]: pipeline_para_               :%d\n",  __FUNCTION__,  __LINE__, getpid(), pipeline_para_               );
    printf("[%s:%d:pid%d]: stream_                      :%p\n",  __FUNCTION__,  __LINE__, getpid(), stream_                      );
    printf("[%s:%d:pid%d]: cublas_wrapper_              :%d\n",  __FUNCTION__,  __LINE__, getpid(), cublas_wrapper_              );
    printf("[%s:%d:pid%d]: allocator_                   :%d\n",  __FUNCTION__,  __LINE__, getpid(), allocator_                   );
    printf("[%s:%d:pid%d]: is_free_buffer_after_forward_:%d\n",  __FUNCTION__,  __LINE__, getpid(), is_free_buffer_after_forward_);
    printf("[%s:%d:pid%d]: sparse_                      :%d\n",  __FUNCTION__,  __LINE__, getpid(), sparse_                      );
    printf("[%s:%d:pid%d]: int8_mode_                   :%d\n",  __FUNCTION__,  __LINE__, getpid(), int8_mode_                   );
    printf("[%s:%d:pid%d]: custom_all_reduce_comm_      :%d\n",  __FUNCTION__,  __LINE__, getpid(), custom_all_reduce_comm_      );
    printf("[%s:%d:pid%d]: enable_custom_all_reduce_    :%d\n",  __FUNCTION__,  __LINE__, getpid(), enable_custom_all_reduce_    );
    printf("*******************************************************************************************\n");
    gpt_decoder_ = new ParallelGptDecoder<T>(0,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             num_layer_,
                                             expert_num_,
                                             moe_k_,
                                             moe_layer_index_,
                                             layernorm_eps_,
                                             gpt_variant_params_,
                                             tensor_para_,
                                             pipeline_para_,
                                             stream_,
                                             cublas_wrapper_,
                                             allocator_,
                                             is_free_buffer_after_forward_,
                                             sparse_,
                                             int8_mode_,
                                             custom_all_reduce_comm_,
                                             enable_custom_all_reduce_);

    printf("*******************************************************************************************\n");
    printf("DynamicDecodeLayer参数:\n");
    printf("[%s:%d:pid%d]: vocab_size_                  :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)vocab_size_);
    printf("[%s:%d:pid%d]: vocab_size_padded_           :%d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)vocab_size_padded_);
    printf("[%s:%d:pid%d]: end_id                       :%d\n",  __FUNCTION__,  __LINE__, getpid(), 0);
    printf("[%s:%d:pid%d]: stream_                      :%p\n",  __FUNCTION__,  __LINE__, getpid(), stream_);
    printf("[%s:%d:pid%d]: cublas_wrapper_              :%p\n",  __FUNCTION__,  __LINE__, getpid(), cublas_wrapper_);
    printf("[%s:%d:pid%d]: allocator_                   :%p\n",  __FUNCTION__,  __LINE__, getpid(), allocator_);
    printf("[%s:%d:pid%d]: is_free_buffer_after_forward_:%d\n",  __FUNCTION__,  __LINE__, getpid(), is_free_buffer_after_forward_);
    printf("[%s:%d:pid%d]: cuda_device_prop_            :%p\n",  __FUNCTION__,  __LINE__, getpid(), cuda_device_prop_);
    printf("*******************************************************************************************\n");
    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void ParallelGpt<T>::allocateBuffer(size_t batch_size,//1
                                    size_t beam_width,//1
                                    size_t max_session_len,//40
                                    size_t memory_len,//40
                                    size_t max_input_len,//8+0
                                    bool   is_return_context_cum_log_probs)//0
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam = batch_size * beam_width;//1*1=1
    const size_t self_cache_size =
        (num_layer_ / pipeline_para_.world_size_) * batchxbeam * memory_len * hidden_units_ / tensor_para_.world_size_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    printf("[%s:%d:pid%d]: [input_attention_mask_]  size:%d, batchxbeam:%d, max_input_len:%d max_input_len:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * max_input_len * max_input_len,
        batchxbeam, max_input_len, max_input_len);
    printf("[%s:%d:pid%d]: [decoder_input_buf_]  size:%d, batchxbeam:%d,hidden_units_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * hidden_units_,
        batchxbeam, hidden_units_);
    printf("[%s:%d:pid%d]: [decoder_normed_input_buf_]  size:%d, batchxbeam:%d,hidden_units_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * hidden_units_,
        batchxbeam, hidden_units_);
    printf("[%s:%d:pid%d]: [decoder_output_buf_]  size:%d, batchxbeam:%d,hidden_units_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * hidden_units_,
        batchxbeam, hidden_units_);

    printf("[%s:%d:pid%d]: [normed_decoder_output_buf_]  size:%d, batchxbeam:%d,hidden_units_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * hidden_units_,
        batchxbeam, hidden_units_);
    printf("[%s:%d:pid%d]: [logits_buf_]  size:%d, batchxbeam:%d vocab_size_padded_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(float) * batchxbeam * vocab_size_padded_,
        batchxbeam, vocab_size_padded_);
    printf("[%s:%d:pid%d]: [cum_log_probs_]  size:%d, batchxbeam:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(float) * batchxbeam,
        batchxbeam);
    printf("[%s:%d:pid%d]: [finished_buf_]  size:%d, batchxbeam:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(bool) * batchxbeam,
        batchxbeam);
    printf("[%s:%d:pid%d]: [h_finished_buf_]  size:%d, batchxbeam:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(bool) * batchxbeam,
        batchxbeam);
    printf("[%s:%d:pid%d]: [sequence_lengths_]  size:%d, batchxbeam:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(int) * batchxbeam,
        batchxbeam);

    input_attention_mask_ = (T*)(allocator_->reMalloc(
        input_attention_mask_, sizeof(T) * batchxbeam * max_input_len * max_input_len, false));
    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_normed_input_buf_ =
        (T*)(allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    logits_buf_ = (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ =
        (float*)(allocator_->reMalloc(nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_    = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_     = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_   = new bool[batchxbeam];
    sequence_lengths_ = (int*)(allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false));

    printf("[%s:%d:pid%d]: [key_cache_]  size:%d, self_cache_size:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * self_cache_size * 1,
        self_cache_size);
    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    printf("[%s:%d:pid%d]: [value_cache_]  size:%d, self_cache_size:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * self_cache_size * 1,
        self_cache_size);
    value_cache_ = key_cache_ + self_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] =
            (int*)(allocator_->reMalloc(cache_indirections_[0], sizeof(int) * batchxbeam * memory_len * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * memory_len;
    }

    printf("[%s:%d:pid%d]: [tiled_input_ids_buf_]  size:%d, max_session_len:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(int) * batchxbeam * max_session_len,
        max_session_len);
    printf("[%s:%d:pid%d]: [tiled_input_lengths_buf_]  size:%d, batchxbeam:%d \n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(int) * batchxbeam,
        batchxbeam);
    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));

    // prompt_learning weight batch ptrs
    prompt_learning_weight_batch_ =
        (const T**)(allocator_->reMalloc(prompt_learning_weight_batch_, sizeof(T*) * batchxbeam, false));
    tiled_prompt_lengths_buf_ =
        (int*)(allocator_->reMalloc(tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_   = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    printf("[%s:%d:pid%d]: [transposed_output_ids_buf_]  size:%d, batchxbeam:%d max_session_len:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(int) * batchxbeam * max_session_len,
        batchxbeam,max_session_len);
    transposed_output_ids_buf_ =
        (int*)(allocator_->reMalloc(transposed_output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    output_ids_buf_ = (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    parent_ids_buf_ = (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * max_session_len, true));
    printf("[%s:%d:pid%d]: [seq_limit_len_]  size:%d, batch_size:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(uint32_t) * batch_size,
        batch_size);
    seq_limit_len_  = (uint32_t*)(allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false));
    printf("[%s:%d:pid%d]: [masked_tokens_]  size:%d, batchxbeam:%d memory_len:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(bool) * batchxbeam * memory_len,
        batchxbeam, memory_len);
    masked_tokens_  = (bool*)(allocator_->reMalloc(masked_tokens_, sizeof(bool) * batchxbeam * memory_len, true));

    printf("[%s:%d:pid%d]: [context_decoder_input_buf_]  size:%d, batchxbeam:%d max_input_len:%d hidden_units_:%d\n",  __FUNCTION__,  __LINE__, getpid(), 
        sizeof(T) * batchxbeam * max_input_len * hidden_units_,
        batchxbeam, max_input_len, hidden_units_);
    context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_session_len, false));

    if (gpt_variant_params_.has_pre_decoder_layernorm) {
        context_decoder_normed_input_buf_ = (T*)allocator_->reMalloc(
            context_decoder_normed_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false);
        decoder_normed_input_buf_ =
            (T*)allocator_->reMalloc(decoder_normed_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false);
    }

    if (gpt_variant_params_.use_attention_linear_bias) {
        linear_bias_slopes_ = (T*)(allocator_->reMalloc(linear_bias_slopes_, sizeof(T) * head_num_, false));
    }

    if (is_return_context_cum_log_probs) {
        lp_normed_decoder_output_buf_ = (T*)allocator_->reMalloc(
            lp_normed_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_);
        lp_logits_buf_      = (float*)allocator_->reMalloc(lp_logits_buf_,
                                                      sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_nccl_logits_buf_ = (float*)allocator_->reMalloc(
            lp_nccl_logits_buf_, sizeof(float) * batchxbeam * max_input_len * vocab_size_padded_);
        lp_logprob_buf_ = (float*)allocator_->reMalloc(lp_logprob_buf_, sizeof(float) * batchxbeam * max_input_len);
    }
    if (shared_contexts_ratio_ > 0.0f) {
        shared_contexts_idx_  = (int*)allocator_->reMalloc(shared_contexts_idx_, 3 * batch_size * sizeof(int), false);
        batch_to_compact_idx_ = shared_contexts_idx_ + batch_size;
        compact_idx_          = shared_contexts_idx_ + 2 * batch_size;
        compact_size_         = (int*)allocator_->reMalloc(compact_size_, sizeof(int), false);
    }

    if (generation_should_stop_ == nullptr) {
        cudaMallocHost(&generation_should_stop_, 1 * sizeof(bool));
    }
    tiled_total_padding_count_ =
        (int*)allocator_->reMalloc(tiled_total_padding_count_, batchxbeam * sizeof(int), false);

    is_allocate_buffer_ = true;
}

template<typename T>
void ParallelGpt<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void**)(&padded_embedding_kernel_));
        }

        allocator_->free((void**)(&input_attention_mask_));
        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&normed_decoder_output_buf_));
        allocator_->free((void**)(&logits_buf_));
        allocator_->free((void**)(&nccl_logits_buf_));
        allocator_->free((void**)(&cum_log_probs_));
        allocator_->free((void**)(&finished_buf_));
        delete[] h_finished_buf_;
        allocator_->free((void**)(&sequence_lengths_));

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&tiled_input_ids_buf_));
        allocator_->free((void**)(&tiled_input_lengths_buf_));

        allocator_->free((void**)(&prompt_learning_weight_batch_));
        allocator_->free((void**)(&tiled_prompt_lengths_buf_));

        allocator_->free((void**)(&transposed_output_ids_buf_));
        allocator_->free((void**)(&output_ids_buf_));
        allocator_->free((void**)(&parent_ids_buf_));
        allocator_->free((void**)(&masked_tokens_));

        allocator_->free((void**)(&seq_limit_len_));

        allocator_->free((void**)(&start_ids_buf_));
        allocator_->free((void**)(&end_ids_buf_));

        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));
        allocator_->free((void**)(&output_log_probs_buf_));

        if (gpt_variant_params_.has_pre_decoder_layernorm) {
            allocator_->free((void**)(&context_decoder_normed_input_buf_));
            allocator_->free((void**)(&decoder_normed_input_buf_));
        }
        if (gpt_variant_params_.use_attention_linear_bias) {
            allocator_->free((void**)(&linear_bias_slopes_));
        }

        allocator_->free((void**)(&lp_normed_decoder_output_buf_));
        allocator_->free((void**)(&lp_logits_buf_));
        allocator_->free((void**)(&lp_nccl_logits_buf_));
        allocator_->free((void**)(&lp_logprob_buf_));

        cudaFreeHost(generation_should_stop_);

        if (shared_contexts_ratio_ > 0.0f) {
            allocator_->free((void**)(&shared_contexts_idx_));
            allocator_->free((void**)(&compact_size_));
        }
        allocator_->free((void**)(&tiled_total_padding_count_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
ParallelGpt<T>::ParallelGpt(size_t                              max_batch_size,
                            size_t                              max_seq_len,
                            size_t                              max_input_len,
                            size_t                              beam_width,
                            size_t                              head_num,
                            size_t                              size_per_head,
                            size_t                              inter_size,
                            size_t                              num_layer,
                            size_t                              expert_num,
                            size_t                              moe_k,
                            std::vector<int64_t>                moe_layer_index,
                            size_t                              vocab_size,
                            int                                 start_id,
                            int                                 end_id,
                            int                                 prompt_learning_start_id,
                            PromptLearningType                  prompt_learning_type,
                            gptVariantParams                    gpt_variant_params,
                            float                               beam_search_diversity_rate,
                            size_t                              top_k,
                            float                               top_p,
                            unsigned long long                  random_seed,
                            float                               temperature,
                            float                               len_penalty,
                            float                               repetition_penalty,
                            NcclParam                           tensor_para,
                            NcclParam                           pipeline_para,
                            cudaStream_t                        stream,
                            cublasMMWrapper*                    cublas_wrapper,
                            IAllocator*                         allocator,
                            bool                                is_free_buffer_after_forward,
                            cudaDeviceProp*                     cuda_device_prop,
                            AttentionType                       attention_type,
                            bool                                sparse,
                            int                                 int8_mode,
                            std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                            int                                 enable_custom_all_reduce,
                            float                               shared_contexts_ratio):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop, sparse),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    expert_num_(expert_num),
    moe_k_(moe_k),
    moe_layer_index_(moe_layer_index),
    vocab_size_(vocab_size),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    layernorm_eps_(gpt_variant_params.layernorm_eps),
    gpt_variant_params_(gpt_variant_params),
    beam_search_diversity_rate_(beam_search_diversity_rate),
    hidden_units_(head_num_ * size_per_head),
    top_k_(top_k),
    top_p_(top_p),
    random_seed_(random_seed),
    temperature_(temperature),
    len_penalty_(len_penalty),
    repetition_penalty_(repetition_penalty),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    local_head_num_(head_num / tensor_para.world_size_),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    shared_contexts_ratio_(shared_contexts_ratio)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
ParallelGpt<T>::ParallelGpt(ParallelGpt<T> const& gpt):
    BaseLayer(gpt),
    head_num_(gpt.head_num_),
    size_per_head_(gpt.size_per_head_),
    inter_size_(gpt.inter_size_),
    num_layer_(gpt.num_layer_),
    expert_num_(gpt.expert_num_),
    moe_k_(gpt.moe_k_),
    moe_layer_index_(gpt.moe_layer_index_),
    vocab_size_(gpt.vocab_size_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    prompt_learning_start_id_(gpt.prompt_learning_start_id_),
    prompt_learning_type_(gpt.prompt_learning_type_),
    beam_search_diversity_rate_(gpt.beam_search_diversity_rate_),
    layernorm_eps_(gpt.gpt_variant_params_.layernorm_eps),
    gpt_variant_params_(gpt.gpt_variant_params_),
    hidden_units_(gpt.hidden_units_),
    top_k_(gpt.top_k_),
    top_p_(gpt.top_p_),
    random_seed_(gpt.random_seed_),
    temperature_(gpt.temperature_),
    len_penalty_(gpt.len_penalty_),
    repetition_penalty_(gpt.repetition_penalty_),
    tensor_para_(gpt.tensor_para_),
    pipeline_para_(gpt.pipeline_para_),
    local_head_num_(gpt.local_head_num_),
    vocab_size_padded_(gpt.vocab_size_padded_),
    attention_type_(gpt.attention_type_),
    int8_mode_(gpt.int8_mode_),
    custom_all_reduce_comm_(gpt.custom_all_reduce_comm_),
    enable_custom_all_reduce_(gpt.enable_custom_all_reduce_),
    shared_contexts_ratio_(gpt.shared_contexts_ratio_)
{
    initialize();
}

template<typename T>
ParallelGpt<T>::~ParallelGpt()
{
    delete gpt_decoder_;
    delete gpt_context_decoder_;
    delete dynamic_decode_layer_;
    freeBuffer();
}

template<typename T>
void ParallelGpt<T>::computeContextCumLogProbs(float*                      cum_log_probs,
                                               const T*                    context_decoder_outputs,
                                               const int*                  input_ids,
                                               const int*                  input_lengths,
                                               const size_t                batch_size,
                                               const size_t                beam_width,
                                               const size_t                max_input_length,
                                               const ParallelGptWeight<T>* gpt_weights)
{
    // Compute the log probabilties of prompt inputs.
    //
    // cum_log_probs [batch_size, beam_width]
    // context_decoder_outputs [batch_size * beam_width, max_input_length, hidden_units]
    // input_ids [batch_size * beam_width, max_input_length]; input ids.
    // input_lengths [batch_size, beam_width]; input lengths.
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t batchxbeam      = batch_size * beam_width;
    const size_t n_hidden_states = batchxbeam * max_input_length;

    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        // normed decoder output [batch_size * beam_width, max_input_length, hidden_units_]
        invokeGeneralLayerNorm(lp_normed_decoder_output_buf_,
                               context_decoder_outputs,
                               gpt_weights->post_decoder_layernorm.gamma,
                               gpt_weights->post_decoder_layernorm.beta,
                               layernorm_eps_,
                               n_hidden_states,
                               hidden_units_,
                               (float*)nullptr,
                               0,
                               stream_);
        sync_check_cuda_error();
        if (tensor_para_.world_size_ == 1) {
            float alpha = 1.0f;
            float beta  = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  vocab_size_padded_,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  &beta,
                                  lp_logits_buf_,
                                  CUDA_R_32F,
                                  vocab_size_padded_, /* n */
                                  CUDA_R_32F,
                                  cublasGemmAlgo_t(-1));
            sync_check_cuda_error();
        }
        else {
            FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
            const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
            float     alpha            = 1.0f;
            float     beta             = 0.0f;
            cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  local_vocab_size,  // n
                                  n_hidden_states,
                                  hidden_units_,  // k
                                  &alpha,
                                  padded_embedding_kernel_ptr_ + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  lp_normed_decoder_output_buf_,
                                  sizeof(T) == 2 ? CUDA_R_16F : CUDA_R_32F,
                                  hidden_units_,  // k
                                  &beta,
                                  lp_nccl_logits_buf_ + tensor_para_.rank_ * n_hidden_states * local_vocab_size,
                                  CUDA_R_32F,
                                  local_vocab_size, /* n */
                                  CUDA_R_32F,
                                  cublasGemmAlgo_t(-1));
            sync_check_cuda_error();
            ftNcclAllGather(lp_nccl_logits_buf_,
                            lp_nccl_logits_buf_,
                            n_hidden_states * local_vocab_size,
                            tensor_para_.rank_,
                            tensor_para_,
                            stream_);
            check_cuda_error(cudaStreamSynchronize(stream_));
            sync_check_cuda_error();

            invokeTransposeAxis01(lp_logits_buf_,
                                  lp_nccl_logits_buf_,
                                  tensor_para_.world_size_,
                                  n_hidden_states,
                                  local_vocab_size,
                                  stream_);
            sync_check_cuda_error();
        }
    }

    invokeLogProbFromLogits(cum_log_probs,
                            lp_logits_buf_,
                            input_ids,
                            input_lengths,
                            max_input_length,
                            batchxbeam,
                            vocab_size_,
                            vocab_size_padded_,
                            lp_logprob_buf_,
                            sizeof(float) * batchxbeam * max_input_length,
                            stream_,
                            true);
    sync_check_cuda_error();
}

template<typename T>
void ParallelGpt<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void ParallelGpt<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void ParallelGpt<T>::forward(std::vector<Tensor>*        output_tensors,
                             const std::vector<Tensor>*  input_tensors,
                             const ParallelGptWeight<T>* gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      max_output_seq_len [1] on cpu

    // output_tensors:
    //      output_ids [batch_size, beam, max_output_seq_len]
    //      sequence_length [batch_size, beam]
    //      output_log_probs [batch_size, beam, request_output_seq_len], must be float*.
    //          It leads to additional computing cost. If we don't need this result, please put nullptr
    //      cum_log_probs [batch_size, beam], must be float*, optional
    //          The cumulative log probability of generated sequences. It leads additional computing cost.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    std::unordered_map<std::string, Tensor> input_tensors_map{{"input_ids", input_tensors->at(0)},
                                                              {"input_lengths", input_tensors->at(1)},
                                                              {"max_output_seq_len", input_tensors->at(2)}};
    input_tensors_map.insert({"random_seed", {MEMORY_CPU, TYPE_INT32, {1}, &random_seed_}});
    input_tensors_map.insert({"runtime_top_k", {MEMORY_CPU, TYPE_UINT32, {1}, &top_k_}});
    input_tensors_map.insert({"runtime_top_p", {MEMORY_CPU, TYPE_FP32, {1}, &top_p_}});

    std::unordered_map<std::string, Tensor> output_tensors_map{{"output_ids", output_tensors->at(0)},
                                                               {"sequence_length", output_tensors->at(1)},
                                                               {"output_log_probs", output_tensors->at(2)}};
    if (output_tensors->size() > 3) {
        output_tensors_map.insert({"cum_log_probs", output_tensors->at(4)});
    }
    //运行forward
    forward(&output_tensors_map, &input_tensors_map, gpt_weights);
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

template<typename T>
void ParallelGpt<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                             const std::unordered_map<std::string, Tensor>* input_tensors,
                             const ParallelGptWeight<T>*                    gpt_weights)
{
    printf("------forward begin... ... \n",  __FUNCTION__,  __LINE__, getpid());
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      input_lengths_h [batch_size] on cpu, optional
    //      prompt_learning_task_name_ids [batch_size] on cpu
    //      output_seq_len [batch_size] on cpu
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      presence_penalty [1] or [batch_size] on cpu, optional, float.
    //          Only one of repetition and presence penalties is allowed.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_lengths_h [batch_size], cpu, optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      request_prompt_type [batch_size], int, optional
    //      is_return_context_cum_log_probs [1] on cpu, bool, optional
    //      session_len [1] on cpu, uint32, optional
    //      memory_len [1] on cpu, uint32, optional
    //      continue_gen [1] on cpu, bool, optional
    //      is_return_context_embeddings [1] on cpu, bool, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      sequence_length [batch_size, beam_width]
    //      response_input_lengths [batch_size, beam_width], optional
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam_width], must be float*. optional.
    //          The cumulative log probability of generated sequences. It may lead to additional computing cost.
    //      context_embeddings [batch_size, hidden_units], must be float*, optional

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK(output_tensors->at("sequence_length").shape.size() == 2);
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    // Used when inputs do not contain random_seed
    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];
    FT_CHECK_WITH_INFO(output_tensors->count("cum_log_probs") == 0
                           || output_tensors->at("cum_log_probs").size() == batch_size * beam_width,
                       "The shape of cum_log_probs should match with batch_size x beam_width if provided.");
    int max_input_length = input_tensors->at("input_ids").shape[1];

    bool       continue_gen = input_tensors->find("continue_gen") != input_tensors->end() ?
                                  input_tensors->at("continue_gen").getVal<bool>() :
                                  false; //交互式生成使能
    const bool is_return_context_embeddings =
        input_tensors->find("is_return_context_embeddings") != input_tensors->end()
        && input_tensors->at("is_return_context_embeddings").getVal<bool>();
    if (is_return_context_embeddings) {
        FT_CHECK_WITH_INFO(output_tensors->find("context_embeddings") != output_tensors->end(),
                           "When requesting context embeddings, a context embeddings output tensors must be provided");
    }

    const int initial_step    = continue_gen ? step_ : 0;
    int       max_context_len = max_input_length + initial_step;

    // NOTE: the input already contains the p/prompt-tunning tokens ids for p/prompt tuning task
    // prompt_learning_task_name_ids are used by both p/prompt-tunning and prefix_prompt task
    const int* prompt_learning_task_name_ids =
        input_tensors->count("prompt_learning_task_name_ids") ?
            input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
            nullptr;

    FT_CHECK_WITH_INFO(
        !(prompt_learning_task_name_ids != nullptr
          && (prompt_learning_type_ == PromptLearningType::no_prompt
              || prompt_learning_type_ == PromptLearningType::soft_prompt)),
        "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int                valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");

    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        if (prompt_learning_task_name_ids != nullptr) {
            FT_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
        }
    }
    else if (valid_prompt_inputs > 0) {
        FT_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        FT_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // whether or not use prompt embeddings from the request.
    // If true, staticlly loaded prompts weights during model loading and task name ids will be ignored
    bool use_request_p_prompt_embedding = request_prompt_type == PromptLearningType::p_prompt_tuning;
    int  max_request_p_prompt_length =
        use_request_p_prompt_embedding ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    // p_prompt tuning: input and prompt are concatnenated (not separate),
    const uint32_t* input_lengths_h = input_tensors->count("input_lengths_h") ?
                                          input_tensors->at("input_lengths_h").getPtr<const uint32_t>() :
                                          nullptr;

    size_t max_input_without_prompt_length = max_context_len;
    if (use_request_p_prompt_embedding && input_lengths_h != nullptr
        && input_tensors->count("request_prompt_lengths_h")) {

        const uint32_t* request_prompt_lengths_h =
            input_tensors->at("request_prompt_lengths_h").getPtr<const uint32_t>();
        max_input_without_prompt_length = input_lengths_h[0] - request_prompt_lengths_h[0];
        for (int bs_id = 1; bs_id < batch_size; ++bs_id) {
            max_input_without_prompt_length = std::max(size_t(input_lengths_h[bs_id] - request_prompt_lengths_h[bs_id]),
                                                       max_input_without_prompt_length);
        }
    }

    has_prefix_prompt_ =
        (prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::prefix_prompt);
    has_p_prompt_tuning_ =
        prompt_learning_task_name_ids != nullptr && prompt_learning_type_ == PromptLearningType::p_prompt_tuning
        || use_request_p_prompt_embedding;
    bool use_loaded_p_prompt_embedding = has_p_prompt_tuning_ && !use_request_p_prompt_embedding;
    has_prefix_soft_prompt_            = request_prompt_type == PromptLearningType::soft_prompt;

    //hard prompt就是由具体的中文或英文词汇组成提示，它是人工可读的提示
    //soft prompt提示是在向量空间优化出来的提示，从一个hard prompt开始（初始化）通过梯度搜索之类的方式进行优化，不改变原始的提示向量的数量和位置，在它的空间进行搜索。
    // NOTE: soft prompt
    FT_CHECK_WITH_INFO(!(has_prefix_soft_prompt_ && continue_gen),
                       "Interactive Generations cannot work with prefix_soft_prompt !");
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t gen_len          = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;

    size_t session_len = 0;
    if (continue_gen) {
        session_len = session_len_;  // Record the size of allocated buffer in previous round.
    }
    else if (input_tensors->find("session_len") != input_tensors->end()) {
        session_len = input_tensors->at("session_len").getVal<uint32_t>();  // Use for allocate buffer in first round.
    }
    else {
        session_len = gen_len;  // When the interactive generation mode is disabled.
    }
    session_len_ = session_len;
    FT_CHECK_WITH_INFO(
        gen_len + initial_step <= session_len,
        fmtstr("Session size too low (%d) vs. total output size (%d)", session_len, gen_len + initial_step));
    size_t memory_len = 0;
    if (continue_gen) {
        memory_len = memory_len_;  // Record the size of allocated buffer in previous round.
    }
    else if (input_tensors->find("memory_len") != input_tensors->end()) {
        memory_len = input_tensors->at("memory_len").getVal<uint32_t>();  // Use for allocate buffer in first round.
    }
    else {
        memory_len = session_len;  // When the interactive generation mode is disabled.
    }
    memory_len_ = memory_len;
    /* TODO: could remove this constraint by changing how context decoder operates */
    FT_CHECK_WITH_INFO(max_input_length <= memory_len,
                       fmtstr("Memory size too low (%d) vs. input length (%d)", memory_len, max_input_length));

    if (memory_len < session_len) {
        FT_LOG_WARNING("memory_len (%d) is less than session_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       memory_len,
                       session_len);
    }
    else if (memory_len > session_len) {
        FT_LOG_WARNING("memory_len (%d) is larger than session_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller memory_len.",
                       memory_len,
                       session_len);
    }

    if (gpt_variant_params_.has_positional_encoding && session_len_ > gpt_weights->getMaxSeqLen()) {
        FT_LOG_ERROR("The session_len_ (%d) of request is longer than max_seq_len (%d) of embedding table."
                     " This is a invalid input. Setting the session_len_ to %d.",
                     session_len_,
                     gpt_weights->getMaxSeqLen(),
                     gpt_weights->getMaxSeqLen());
        session_len_ = gpt_weights->getMaxSeqLen();
    }

    const bool is_return_context_cum_log_probs = input_tensors->count("is_return_context_cum_log_probs") > 0
                                                 && input_tensors->at("is_return_context_cum_log_probs").getVal<bool>();
    if (is_return_context_cum_log_probs) {
        FT_CHECK_WITH_INFO(output_tensors->count("cum_log_probs")
                               && output_tensors->at("cum_log_probs").data != nullptr,
                           "`cum_log_probs` must be provided in `output_tensors` in order to enable "
                           "the cumulative log probability computation of input contexts.");
    }

    printf("**************************************************************************************\n");
    printf("[%s:%d-pid:%d]:  参数数据信息： \n", __FUNCTION__,  __LINE__, getpid(),  continue_gen);
    printf("[%s:%d:pid%d]: input_tensors->size:                               %d\n",  __FUNCTION__,  __LINE__, getpid(), input_tensors->size());
    printf("[%s:%d:pid%d]: output_tensors->size:                              %d\n",  __FUNCTION__,  __LINE__, getpid(), output_tensors->size());
    printf("[%s:%d:pid%d]: input_tensors->at(input_ids).shape.size():         %d\n",  __FUNCTION__,  __LINE__, getpid(), input_tensors->at("input_ids").shape.size());
    printf("[%s:%d:pid%d]: input_tensors->at(input_lengths).shape.size():     %d\n",  __FUNCTION__,  __LINE__, getpid(), input_tensors->at("input_lengths").shape.size());
    printf("[%s:%d:pid%d]: input_tensors->at(output_seq_len).shape.size():    %d\n",  __FUNCTION__,  __LINE__, getpid(), input_tensors->at("output_seq_len").shape.size());
    printf("[%s:%d:pid%d]: output_tensors->at(output_ids).shape.size():       %d\n",  __FUNCTION__,  __LINE__, getpid(), output_tensors->at("output_ids").shape.size());
    printf("[%s:%d:pid%d]: batch_size:                                        %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)batch_size);
    printf("[%s:%d:pid%d]: beam_width:                                        %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)beam_width);
    printf("[%s:%d:pid%d]: max_input_length:                                  %d\n",  __FUNCTION__,  __LINE__, getpid(), max_input_length);
    printf("[%s:%d:pid%d]: continue_gen:                                      %d\n",  __FUNCTION__,  __LINE__, getpid(), continue_gen);
    printf("[%s:%d:pid%d]: is_return_context_embeddings:                      %d\n",  __FUNCTION__,  __LINE__, getpid(), is_return_context_embeddings);
    printf("[%s:%d:pid%d]: initial_step:                                      %d\n",  __FUNCTION__,  __LINE__, getpid(), initial_step);
    printf("[%s:%d:pid%d]: step_:                                             %d\n",  __FUNCTION__,  __LINE__, getpid(), step_);
    printf("[%s:%d:pid%d]: max_context_len:                                   %d\n",  __FUNCTION__,  __LINE__, getpid(), max_context_len);
    printf("[%s:%d:pid%d]: valid_prompt_inputs:                               %d\n",  __FUNCTION__,  __LINE__, getpid(), valid_prompt_inputs);
    printf("[%s:%d:pid%d]: request_prompt_type:                               %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)request_prompt_type);
    printf("[%s:%d:pid%d]: use_request_p_prompt_embedding:                    %d\n",  __FUNCTION__,  __LINE__, getpid(), use_request_p_prompt_embedding);
    printf("[%s:%d:pid%d]: max_request_p_prompt_length:                       %d\n",  __FUNCTION__,  __LINE__, getpid(), max_request_p_prompt_length);
    printf("[%s:%d:pid%d]: max_input_without_prompt_length:                   %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)max_input_without_prompt_length);
    printf("[%s:%d:pid%d]: use_loaded_p_prompt_embedding:                     %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)use_loaded_p_prompt_embedding);
    printf("[%s:%d:pid%d]: max_prefix_soft_prompt_length:                     %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)max_prefix_soft_prompt_length);
    printf("[%s:%d:pid%d]: limit_len_offset:                                  %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)limit_len_offset);
    printf("[%s:%d:pid%d]: gen_len（output_seq_len+limit_len_offset）:        %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gen_len);
    printf("[%s:%d:pid%d]: input_tensors->at(output_seq_len).max<uint32_t>(): %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)input_tensors->at("output_seq_len").max<uint32_t>());
    printf("[%s:%d:pid%d]: session_len:                                       %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)session_len);
    printf("[%s:%d:pid%d]: session_len_:                                      %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)session_len_);
    printf("[%s:%d:pid%d]: memory_len:                                        %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)memory_len);

    printf("[%s:%d:pid%d]: use_attention_linear_bias:                         %d\n",  __FUNCTION__,  __LINE__, getpid(), (int32_t)gpt_variant_params_.use_attention_linear_bias);
    printf("[%s:%d:pid%d]: use_shared_contexts:                               %d\n",  __FUNCTION__,  __LINE__, getpid(), (shared_contexts_ratio_ > 0.0f) && (max_input_length >= 1) && (batch_size > 1));
    printf("****************************************************************************************\n");

    PUSH_RANGE("buffer allocation");
    if (!continue_gen) {
        printf("[%s:%d-pid:%d]:  <allocateBuffer>: [buffer allocation] batch_size=%d, beam_width=%d, session_len=%d, memory_len=%d; max_input_length=%d, max_prefix_soft_prompt_length=%d;is_return_context_cum_log_probs=%d\n", 
            __FUNCTION__,  __LINE__, getpid(),  
            (int32_t)batch_size,
            (int32_t)beam_width,
            (int32_t)session_len,
            (int32_t)memory_len,
            (int32_t)max_input_length,
            (int32_t)max_prefix_soft_prompt_length,
            (int32_t)is_return_context_cum_log_probs);
        allocateBuffer(batch_size,
                       beam_width,
                       session_len,
                       memory_len,
                       max_input_length + max_prefix_soft_prompt_length,
                       is_return_context_cum_log_probs);
        sync_check_cuda_error();
    }

    printf("[%s:%d-pid:%d]:  <setSeqLimitLen>: seq_limit_len_=%p, limit_len_offset=%d, batch_size=%d, input_tensors->at(output_seq_len)=%d shapess=%d/%d %d\n", 
            __FUNCTION__,  __LINE__, getpid(),  
            seq_limit_len_,//buffer长度batch_size[1]
            (int32_t)limit_len_offset,//0
            (int32_t)batch_size,//1
            input_tensors->at("output_seq_len").max<uint32_t>(),//40
            input_tensors->at("output_seq_len").shape[0],
            input_tensors->at("output_seq_len").shape[1],
            input_tensors->at("output_seq_len").size());
    setSeqLimitLen(seq_limit_len_, input_tensors->at("output_seq_len"), limit_len_offset, batch_size);//input_tensors host; seq_limit_len_ device
    POP_RANGE;

    const DataType       data_type      = getTensorType<T>();
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    memory_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {
        num_layer_ / pipeline_para_.world_size_, batch_size * beam_width, local_head_num_, memory_len, size_per_head_};

    {
        PUSH_RANGE("dynamic decode setup");
        TensorMap input_map(*input_tensors);
        printf("[%s:%d-pid:%d]:  <dynamic_decode_layer_->setup>: [dynamic decode setup] batch_size=%d, beam_width=%d\n", 
            __FUNCTION__,  __LINE__, getpid(),  
            (int32_t)batch_size,
            (int32_t)beam_width);
        dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
        handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
        POP_RANGE;
    }

    if (gpt_variant_params_.use_attention_linear_bias) {
        PUSH_RANGE("build alibi slopes");
        printf("[%s:%d-pid:%d]:  <dynamic_decode_layer_->setup>: [build alibi slopes] head_num_=%d, stream_=%p\n", 
            __FUNCTION__,  __LINE__, getpid(),  
            (int32_t)head_num_,
            stream_);
        invokeBuildAlibiSlopes(linear_bias_slopes_, head_num_, stream_);
        POP_RANGE;
    }

    printf("[%s:%d-pid:%d]:  continue_gen=%d, \n", __FUNCTION__,  __LINE__, getpid(),  continue_gen);

    if (continue_gen) {
        PUSH_RANGE("input tiling and init");
        printf("[%s:%d-pid:%d]: <invokeTileGptInputs>: [input tiling and init] batch_size=%d, beam_width=%d, max_input_length=%d\n", 
            __FUNCTION__,  __LINE__, getpid(),
            batch_size,
            beam_width,
            max_input_length);
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);

        printf("[%s:%d-pid:%d]:  <invokePlusScalar>: batch_size=%d, beam_width=%d\n", 
            __FUNCTION__, __LINE__, getpid(),
            batch_size,
            beam_width);
        invokePlusScalar(tiled_input_lengths_buf_, initial_step, batch_size * beam_width, stream_);
        sync_check_cuda_error();

        printf("[%s:%d-pid:%d]: <invokeDecodingInitialize>: \n", __FUNCTION__, __LINE__, getpid());
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 initial_step - 1,
                                 stream_);
        printf("[%s:%d-pid:%d]: <invokeTransposeAxis01> \n", __FUNCTION__, __LINE__, getpid());
        invokeTransposeAxis01(output_ids_buf_ + initial_step * batch_size * beam_width,
                              tiled_input_ids_buf_,
                              batch_size * beam_width,
                              max_input_length,
                              1,
                              stream_);
        POP_RANGE;
    }
    else {
        // TODO(bhsueh) Initilaize them in one kernel
        // initialize the output ids and parent ids
        PUSH_RANGE("initialize output and parent ids");
        printf("[%s:%d-pid:%d]: <cudaMemsetAsync> [initialize output and parent ids] output_ids_buf_/parent_ids_buf_/masked_tokens_/tiled_total_padding_count_/cache_indirections_\n", __FUNCTION__, __LINE__, getpid());
        cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * session_len, stream_);
        cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * session_len, stream_);
        cudaMemsetAsync(masked_tokens_, false, sizeof(bool) * batch_size * beam_width * memory_len, stream_);
        cudaMemsetAsync(tiled_total_padding_count_, 0, sizeof(int) * batch_size * beam_width, stream_);
        if (beam_width > 1) {
            cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * memory_len, stream_);
        }
        sync_check_cuda_error();
        POP_RANGE;

        PUSH_RANGE("padded embedding kernel init");
        if (vocab_size_ == vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = gpt_weights->post_decoder_embedding.kernel;
        }
        else {
            printf("[%s:%d-pid:%d]: <cudaAutoCpy>: [padded embedding kernel init] vocab_size_=%d, hidden_units_=%d, vocab_size_padded_=%d\n",
                __FUNCTION__, __LINE__, getpid(),
                (int32_t)vocab_size_,
                (int32_t)hidden_units_,
                (int32_t)vocab_size_padded_);
            cudaAutoCpy(padded_embedding_kernel_,
                        gpt_weights->post_decoder_embedding.kernel,
                        vocab_size_ * hidden_units_,
                        stream_);
            sync_check_cuda_error();
        }
        POP_RANGE;

        int  compact_size;
        bool use_shared_contexts = (shared_contexts_ratio_ > 0.0f) && (max_input_length >= 1) && (batch_size > 1);
        PUSH_RANGE("find context dups");
        if (use_shared_contexts) {
            printf("[%s:%d-pid:%d]: <invokeFindContextDups>: [find context dups] compact_size_=%d, batch_size=%d, max_input_length=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                compact_size_,
                batch_size,
                max_input_length);
            invokeFindContextDups(shared_contexts_idx_,
                                  batch_to_compact_idx_,
                                  compact_idx_,
                                  compact_size_,
                                  input_tensors->at("input_ids").getPtr<int>(),
                                  batch_size,
                                  max_input_length,
                                  stream_);
            printf("[%s:%d-pid:%d]: <cudaD2Hcpy> : compact_size_=%d, compact_size=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                compact_size_,
                compact_size);
            cudaD2Hcpy(&compact_size, compact_size_, 1);
            use_shared_contexts = compact_size <= shared_contexts_ratio_ * batch_size;
            printf("[%s:%d-pid:%d]: <cudaD2Hcpy> ret: compact_size_=%d, compact_size=%d, use_shared_contexts=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                compact_size_,
                compact_size,
                use_shared_contexts);
            sync_check_cuda_error();
        }
        POP_RANGE;

        // NOTE: p/prompt-tuning process here (lookup prompt embedding tables by task name ids)
        // get p/prompt-tuning weight for each batch --> shape [batch, beam_width]
        // --> ptrs with shape [prompt_len, hidden_size]
        std::vector<const T*> p_prompt_tuning_batch_ptrs;
        std::vector<int>      p_prompt_tuning_lengths;
        PUSH_RANGE("prompt embedding lookup");
        if (use_loaded_p_prompt_embedding) {
            printf("[%s:%d-pid:%d]: <prompt embedding lookup>: [prompt embedding lookup] loop batch_size=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size);
            for (int bs_id = 0; bs_id < batch_size; ++bs_id) {
                int                      task_id              = prompt_learning_task_name_ids[bs_id];
                std::pair<const T*, int> p_prompt_tuning_pair = {};
                bool                     valid_task_name_id   = task_id < gpt_weights->prompt_learning_table.size();
                if (valid_task_name_id) {
                    p_prompt_tuning_pair = gpt_weights->prompt_learning_table.at(task_id);
                }
                else {
                    // don't throw oor in case of model server failing
                    FT_LOG_ERROR("p_prompt_tuning_weights not found for task id: " + std::to_string(task_id)
                                 + "\n return with invalid output tensors");
                    return;
                }
                if (input_lengths_h != nullptr) {
                    if (bs_id == 0) {
                        max_input_without_prompt_length = input_lengths_h[bs_id] - p_prompt_tuning_pair.second;
                    }
                    else {
                        max_input_without_prompt_length =
                            std::max(size_t(input_lengths_h[bs_id] - p_prompt_tuning_pair.second),
                                     max_input_without_prompt_length);
                    }
                }
                for (int bw_id = 0; bw_id < beam_width; ++bw_id) {
                    // only weight ptrs needed here
                    p_prompt_tuning_batch_ptrs.push_back(p_prompt_tuning_pair.first);
                    p_prompt_tuning_lengths.push_back(p_prompt_tuning_pair.second);
                }
            }

            printf("[%s:%d-pid:%d]: <cudaAutoCpy> : prompt_learning_weight_batch_/tiled_prompt_lengths_buf_ batch_size=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size,
                beam_width);
            cudaAutoCpy(
                prompt_learning_weight_batch_, p_prompt_tuning_batch_ptrs.data(), batch_size * beam_width, stream_);

            cudaAutoCpy(tiled_prompt_lengths_buf_, p_prompt_tuning_lengths.data(), batch_size * beam_width, stream_);

            sync_check_cuda_error();
        }
        POP_RANGE;

        // handle first step
        printf("[%s:%d-pid:%d]: <first step> \n", __FUNCTION__, __LINE__, getpid());
        if (has_p_prompt_tuning_ || has_prefix_prompt_ || has_prefix_soft_prompt_ || max_input_length > 1) {
            PUSH_RANGE("input tiling and init");
            printf("[%s:%d-pid:%d]: <invokeTileGptPromptInputs>: [input tiling and init] has_p_prompt_tuning_=%d, has_prefix_prompt_=%d, has_prefix_soft_prompt_=%d, max_input_length=%d, batch_size=%d, beam_width=%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                has_p_prompt_tuning_,
                has_prefix_prompt_,
                has_prefix_soft_prompt_,
                max_input_length,
                batch_size,
                beam_width);
            invokeTileGptPromptInputs(tiled_input_ids_buf_,
                                      tiled_input_lengths_buf_,
                                      use_request_p_prompt_embedding ? tiled_prompt_lengths_buf_ : nullptr,
                                      input_tensors->at("input_ids").getPtr<int>(),
                                      input_tensors->at("input_lengths").getPtr<const int>(),
                                      use_request_p_prompt_embedding ?
                                          input_tensors->at("request_prompt_lengths").getPtr<const int>() :
                                          nullptr,
                                      batch_size,
                                      beam_width,
                                      max_input_length,
                                      stream_);
            sync_check_cuda_error();
            POP_RANGE;

            if (has_prefix_soft_prompt_) {
                PUSH_RANGE("input id embedding lookup");
                inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param;
                param.from_tensor                   = context_decoder_input_buf_;
                param.output_ids                    = output_ids_buf_;
                param.input_lengths                 = tiled_input_lengths_buf_;
                param.embedding_table               = gpt_weights->pre_decoder_embedding_table;
                param.pos_table                     = gpt_weights->position_encoding_table;
                param.prefix_soft_prompt_embedding  = input_tensors->at("request_prompt_embedding").getPtr<float>();
                param.prefix_soft_prompt_lengths    = input_tensors->at("request_prompt_lengths").getPtr<int>();
                param.input_ids                     = tiled_input_ids_buf_;
                param.start_step                    = 1;
                param.max_input_length              = max_input_length;
                param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
                param.batch_size                    = batch_size;
                param.beam_width                    = beam_width;
                param.hidden_units                  = hidden_units_;
                param.stream                        = stream_;

                printf("[%s:%d-pid:%d]: <invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt> [input id embedding lookup]\n", __FUNCTION__, __LINE__, getpid());
                invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(param);
                sync_check_cuda_error();
                POP_RANGE;

                max_input_length += max_prefix_soft_prompt_length;  // view soft_prompt as input
                max_context_len += max_prefix_soft_prompt_length;
            }
            else {
                // NOTE: add prompt embeddings here (for p/prompt tuning)
                PUSH_RANGE("input id embedding lookup");
                pPromptTuningParam<T> prompt_param{
                    use_loaded_p_prompt_embedding ? prompt_learning_weight_batch_ : (const T**)nullptr,
                    prompt_learning_start_id_,
                    max_request_p_prompt_length,
                    use_request_p_prompt_embedding,
                    use_request_p_prompt_embedding ? input_tensors->at("request_prompt_embedding").getPtr<T>() :
                                                     nullptr};
                printf("[%s:%d-pid:%d]: <invokeInputIdsEmbeddingLookupPosEncoding> [input id embedding lookup] start_step:%d max_input_length:%d batch_size:%d hidden_units_:%d\n", 
                    __FUNCTION__, __LINE__, getpid(),
                    1,max_input_length,batch_size * beam_width, hidden_units_);
                {
                    int out_size = 40;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), output_ids_buf_, out_size);
                    printf("invokeInputIdsEmbeddingLookupPosEncoding pre [output_ids_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    }
                    printf("\n");
                }

                {
                    //tiled_input_ids_buf_
                    int out_size = batch_size * beam_width * session_len;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), tiled_input_ids_buf_, out_size);
                    printf("invokeInputIdsEmbeddingLookupPosEncoding pre [tiled_input_ids_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    }
                    printf("\n");
                }

                // {
                //     int out_size = 1024*2;
                //     printf("B-----------------logits[%d] [runSampling]\n", out_size);
                //     std::vector<T> h_buf(out_size);
                //     cudaD2Hcpy(h_buf.data(), gpt_weights->pre_decoder_embedding_table, out_size);
                    
            
                //     float sum_log = 0.0;
                //     float sum_log1 = 0.0;
                //     float sum_log2 = 0.0;
                //     float sum_log1_a = 0.0;
                //     float sum_log2_a = 0.0;
                //     for(int k=0; k<1024; k++) {
                //         sum_log += (float)h_buf[k]* (float)h_buf[k+1024];
                //         sum_log1 += (float)h_buf[k]* (float)h_buf[k];
                //         sum_log2 += (float)h_buf[k+1024]* (float)h_buf[k+1024];
                //         sum_log1_a += (float)h_buf[k];
                //         sum_log2_a += (float)h_buf[k+1024];
                //     }
            
                //     printf("B-----------------logits[%d] [invokeInputIdsEmbeddingLookupPosEncoding] sum_log[%f] 1:%f/%f 2:%f/%f h_buf:\n", 
                //          sum_log, sum_log1,sum_log1_a,sum_log2,sum_log2_a,out_size);
            
                //     // for(int k=0; k<(out_size>1024?100:out_size); k++) {
                //     //     printf("%.2f ", h_buf[k]);
                //     // } 
                //     printf("\n");
                // }

                {
                    int k=100;
                    int out_size = 1024*k;
                    printf("B-----------------logits[%d] [runSampling]\n", out_size);
                    std::vector<T> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), gpt_weights->pre_decoder_embedding_table, out_size);
                    
            
                    float sum_log[100];
                    float sum_log_self[100];
                    for (int n=0; n<k; n++) {
                        sum_log[n] = 0.0;
                        sum_log_self[n]=0.0;
                        for(int k=0; k<1024; k++) {
                            sum_log[n] += (float)h_buf[k]* (float)h_buf[k+1024*n];
                            sum_log_self[n] += (float)h_buf[k+1024*n]* (float)h_buf[k+1024*n];
                        }

                        printf("B-----------------[invokeInputIdsEmbeddingLookupPosEncoding] %d sum_log[%f] self[%f]\n", 
                            n, sum_log[n], sum_log_self[n]);
                    }
            
                    // printf("B-----------------logits[%d] [invokeInputIdsEmbeddingLookupPosEncoding] sum_log[%f] 1:%f/%f 2:%f/%f h_buf:\n", 
                    //      sum_log, sum_log1,sum_log1_a,sum_log2,sum_log2_a,out_size);
            
                    // for(int k=0; k<(out_size>1024?100:out_size); k++) {
                    //     printf("%.2f ", h_buf[k]);
                    // } 
                    printf("\n");
                }

                cudaEvent_t start, end;
                cudaEventCreate(&start);
                cudaEventCreate(&end);
                cudaEventRecord(start);

                invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,//out: [fp16] [batchxbeam, seq_len, hidden_unit]
                                                         output_ids_buf_,//out: [int] [batchxbeam,seq_len]
                                                         gpt_weights->pre_decoder_embedding_table,//in: [fp16]
                                                         gpt_weights->position_encoding_table,//in: [fp16]
                                                         prompt_param,
                                                         tiled_input_ids_buf_, //in: [int] [batchxbeam,seq_len]]
                                                         1, //start_step
                                                         max_input_length, //length
                                                         max_input_length, //max_length
                                                         batch_size * beam_width,
                                                         hidden_units_,
                                                         stream_);
                cudaDeviceSynchronize();
                cudaEventRecord(end);
                cudaEventSynchronize(end);
                float msec, sec;
                cudaEventElapsedTime(&msec, start, end);
                sec = msec / 1000.0;
                cudaEventDestroy(start);
                cudaEventDestroy(end);
                printf("[%s:%d-pid:%d]: <invokeInputIdsEmbeddingLookupPosEncoding> [input id embedding lookup] costtime：%f s\n", 
                    __FUNCTION__, __LINE__, getpid(), sec);

                {
                    int out_size = batch_size * beam_width * max_input_length * hidden_units_;
                    printf("invokeInputIdsEmbeddingLookupPosEncoding [context_decoder_input_buf_] out_size:%d; batch_size:%d beam_width:%d max_input_length:%d hidden_units_:%d\n", 
                        out_size, batch_size, beam_width, max_input_length, hidden_units_);
                    std::vector<T> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), context_decoder_input_buf_, out_size);
                    printf("invokeInputIdsEmbeddingLookupPosEncoding [context_decoder_input_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%f ", (float)h_buf[k]);
                    }
                    printf("\n");
                }
                {
                    int out_size = 40;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), output_ids_buf_, out_size);
                    printf("invokeInputIdsEmbeddingLookupPosEncoding [output_ids_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    }
                    printf("\n");
                }
                sync_check_cuda_error();
                POP_RANGE;
            }

            if (gpt_variant_params_.has_pre_decoder_layernorm) {
                PUSH_RANGE("pre-decoder layernorm");
                printf("[%s:%d-pid:%d]: <invokeGeneralLayerNorm> [pre-decoder layernorm]\n", __FUNCTION__, __LINE__, getpid());
                invokeGeneralLayerNorm(context_decoder_normed_input_buf_,
                                       context_decoder_input_buf_,
                                       gpt_weights->pre_decoder_layernorm.gamma,//
                                       gpt_weights->pre_decoder_layernorm.beta,//
                                       layernorm_eps_,
                                       batch_size * beam_width * max_input_length,
                                       hidden_units_,
                                       (float*)nullptr,
                                       0,
                                       stream_);
                POP_RANGE;
            }
            PUSH_RANGE("build decoder attention mask");
            printf("[%s:%d-pid:%d]: <invokeBuildDecoderAttentionMask> [build decoder attention mask] batch_size:%d max_seq_len:%d max_prompt_length:%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size * beam_width,max_input_length,0);

            {
                int out_size = batch_size * beam_width * max_input_length * max_input_length;
                std::vector<T> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), input_attention_mask_, out_size);
                printf("-------invokeBuildDecoderAttentionMask pre [input_attention_mask_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                }
                printf("\n");
            }
            invokeBuildDecoderAttentionMask(input_attention_mask_,
                                            tiled_input_lengths_buf_,
                                            nullptr,
                                            batch_size * beam_width,
                                            max_input_length,
                                            0,
                                            stream_);
            {
                int out_size = batch_size * beam_width * max_input_length * max_input_length;
                std::vector<T> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), input_attention_mask_, out_size);
                printf("-------invokeBuildDecoderAttentionMask [input_attention_mask_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                }
                printf("\n");
            }
            sync_check_cuda_error();
            POP_RANGE;

            TensorMap decoder_input_tensors(
                {{"decoder_input",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                         gpt_variant_params_.has_pre_decoder_layernorm ? context_decoder_normed_input_buf_ :
                                                                         context_decoder_input_buf_)},
                 {"attention_mask",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, 1, (size_t)max_input_length, (size_t)max_input_length},
                         input_attention_mask_)},
                 {"input_lengths",
                  Tensor(MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_)}});

            printf("[%s:%d-pid:%d]: [in] decoder_input[%d %d %d] attention_mask[%d %d %d %d] input_lengths[%d]\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size * beam_width, (size_t)max_input_length, hidden_units_,
                batch_size * beam_width, 1, (size_t)max_input_length, (size_t)max_input_length,
                batch_size * beam_width);

            if (use_shared_contexts) {
                decoder_input_tensors.insert("compact_idx",
                                             Tensor(MEMORY_GPU, TYPE_INT32, {(size_t)compact_size}, compact_idx_));
                decoder_input_tensors.insert("batch_to_compact_idx",
                                             Tensor(MEMORY_GPU, TYPE_INT32, {batch_size}, batch_to_compact_idx_));
            }
            if (gpt_variant_params_.use_attention_linear_bias) {
                decoder_input_tensors.insert("linear_bias_slopes",
                                             Tensor(MEMORY_GPU,
                                                    data_type,
                                                    {local_head_num_},
                                                    linear_bias_slopes_ + local_head_num_ * tensor_para_.rank_));
            }

            TensorMap decoder_output_tensors(
                {{"decoder_output",
                  Tensor(MEMORY_GPU,
                         data_type,
                         {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                         context_decoder_output_buf_)},
                 {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_shape, key_cache_)},
                 {"value_cache", Tensor(MEMORY_GPU, data_type, self_v_cache_shape, value_cache_)},
                 {"last_token_hidden_units",
                  Tensor(MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_)}});

            printf("[%s:%d-pid:%d]: [out] decoder_output[%d %d %d] key_cache[%d] value_cache[%d] last_token_hidden_units[%d,%d]\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size * beam_width, (size_t)max_input_length, hidden_units_,
                self_k_cache_shape,
                self_v_cache_shape,
                batch_size * beam_width,hidden_units_);

            // 第一个decoder（gpt_context_decoder_）
            printf("[%s:%d-pid:%d]: <gpt_context_decoder_->forward>\n", 
                __FUNCTION__, __LINE__, getpid());
            gpt_context_decoder_->forward(
                &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            {
                //param.step_ids sequence_lengths_
                int out_size = batch_size;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), sequence_lengths_, out_size);
                printf("-----------------sequence_lengths_ [gpt_context_decoder_->forward] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }

            {
                //param.step_ids output_ids_buf_
                int out_size = batch_size*beam_width * session_len;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), output_ids_buf_, out_size);
                printf("-----------------output_ids_buf_ [gpt_context_decoder_->forward] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }

            if (is_return_context_embeddings) {
                PUSH_RANGE("context embedding sum length dim");
                printf("[%s:%d-pid:%d]: <invokeSumLengthDimension> [context embedding sum length dim] batch_size:%d  input_length:%d hidden_dim:%d out context_embeddings\n", 
                    __FUNCTION__, __LINE__, getpid(),
                    batch_size * beam_width,
                    max_input_length,
                    hidden_units_);
                invokeSumLengthDimension(output_tensors->at("context_embeddings").getPtr<float>(),
                                         context_decoder_output_buf_,
                                         batch_size * beam_width,
                                         max_input_length,
                                         hidden_units_,
                                         stream_);
                POP_RANGE;
            }

            PUSH_RANGE("decoding init");
            printf("[%s:%d-pid:%d]: <invokeDecodingInitialize> [decoding init] batch_size:%d beam_width:%d max_input_length:%d\n", 
                __FUNCTION__, __LINE__, getpid(),
                batch_size,
                beam_width,
                max_input_length - 1);
            invokeDecodingInitialize(finished_buf_,//bool* 
                                     sequence_lengths_,//int*
                                     nullptr,//int* word_ids
                                     cum_log_probs_,
                                     start_ids_buf_,//int*  sentence_ids in
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
            {
                //param.step_ids start_ids_buf_
                int out_size = batch_size;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), start_ids_buf_, out_size);
                printf("-----------------3[start_ids_buf_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }

            {
                //param.step_ids sequence_lengths_
                int out_size = batch_size;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), sequence_lengths_, out_size);
                printf("-----------------3[sequence_lengths_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }

            // {
            //     //param.step_ids finished_buf_
            //     int out_size = batch_size * beam_width;
            //     std::vector<bool> h_buf(out_size);
            //     cudaD2Hcpy(h_buf.data(), finished_buf_, out_size);
            //     printf("-----------------3[finished_buf_] h_buf:\n");
            //     for(int k=0; k<out_size; k++) {
            //         printf("%d ", h_buf[k]);
            //     } 
            //     printf("\n");
            // }

            POP_RANGE;

            if (is_return_context_cum_log_probs) {
                PUSH_RANGE("compute context cumulative log probs");
                printf("[%s:%d-pid:%d]: <computeContextCumLogProbs> [compute context cumulative log probs]\n", __FUNCTION__, __LINE__, getpid());
                computeContextCumLogProbs(cum_log_probs_,
                                          context_decoder_output_buf_,
                                          tiled_input_ids_buf_,
                                          tiled_input_lengths_buf_,
                                          batch_size,
                                          beam_width,
                                          (size_t)max_input_length,
                                          gpt_weights);
                POP_RANGE;
            }
            sync_check_cuda_error();
        }
        else if (max_input_length == 0) {
            FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                     && request_prompt_type == PromptLearningType::no_prompt);
            max_input_length++;
            PUSH_RANGE("decoding init");
            printf("[%s:%d-pid:%d]: <invokeDecodingInitialize> [decoding init]\n", __FUNCTION__, __LINE__, getpid());
            invokeDecodingInitialize(finished_buf_,
                                     sequence_lengths_, //查询token长度
                                     output_ids_buf_,   //输出ids缓存
                                     cum_log_probs_,
                                     start_ids_buf_,    //开始ids缓存
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
           {
                //param.step_ids start_ids_buf_
                int out_size = 1;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), start_ids_buf_, out_size);
                printf("-----------------[start_ids_buf_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }
            std::vector<int> h_input_lengths(batch_size * beam_width, 1);
            cudaAutoCpy(tiled_input_lengths_buf_, h_input_lengths.data(), batch_size * beam_width, stream_);
            sync_check_cuda_error();
            POP_RANGE;
        }
        else if (max_input_length == 1) {
            FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                     && request_prompt_type == PromptLearningType::no_prompt);
            PUSH_RANGE("decoding init");
            printf("[%s:%d-pid:%d]: <invokeDecodingInitialize> [decoding init]\n", __FUNCTION__, __LINE__, getpid());
            invokeDecodingInitialize(finished_buf_,
                                     sequence_lengths_,
                                     nullptr,  //word_ids=null
                                     cum_log_probs_,
                                     start_ids_buf_,
                                     batch_size,
                                     beam_width,
                                     max_input_length - 1,
                                     stream_);
            {
                //param.step_ids start_ids_buf_
                int out_size = 1;
                std::vector<int> h_buf(out_size);
                cudaD2Hcpy(h_buf.data(), start_ids_buf_, out_size);
                printf("-----------------2[start_ids_buf_] h_buf:\n");
                for(int k=0; k<out_size; k++) {
                    printf("%d ", h_buf[k]);
                } 
                printf("\n");
            }
            sync_check_cuda_error();
            POP_RANGE;
            PUSH_RANGE("input tiling and init");
            printf("[%s:%d-pid:%d]: <invokeTileGptInputs>: [input tiling and init] \n", __FUNCTION__, __LINE__, getpid());
            invokeTileGptInputs(tiled_input_ids_buf_,
                                tiled_input_lengths_buf_,
                                input_tensors->at("input_ids").getPtr<int>(),
                                input_tensors->at("input_lengths").getPtr<int>(),
                                batch_size,
                                beam_width,
                                max_input_length,
                                stream_);
            sync_check_cuda_error();

            cudaAutoCpy(output_ids_buf_, tiled_input_ids_buf_, batch_size * beam_width, stream_);
            POP_RANGE;
        }
    }

    PUSH_RANGE("mask padding tokens");
    printf("[%s:%d-pid:%d]: <invokeMaskPaddingTokens> [mask padding tokens] memory_len:%d max_input_length:%d initial_step:%d batch_size:%d beam_width:%d\n", 
        __FUNCTION__, __LINE__, getpid(),
        memory_len,
        max_input_length,
        initial_step,
        batch_size,
        beam_width);
    invokeMaskPaddingTokens(masked_tokens_,//out masked_tokens  batch*beam,memory_len
                            input_tensors->at("input_lengths").getPtr<int>(), //in input_lengths
                            memory_len,//40
                            max_input_length,//8
                            initial_step,//0
                            batch_size,//1
                            beam_width,//1
                            stream_);
    POP_RANGE;

    // If continue, we restart from initial_step because last token hasn't been processed in decoder
    const int step_start = continue_gen ? initial_step : max_input_length;

    const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
    FT_CHECK(batch_size % local_batch_size == 0);

    printf("[%s:%d-pid:%d]: <-----loop【output_seq_len】-----> step_start=%d gen_len=%d  local_batch_size=%d/batch_size=%d; world_size_=%d; continue_gen=%d,initial_step=%d,max_input_length=%d\n", 
        __FUNCTION__, __LINE__, getpid(), 
        step_start,
        gen_len, 
        local_batch_size,
        batch_size,
        pipeline_para_.world_size_,
        continue_gen,initial_step,max_input_length);
    for (step_ = step_start; step_ < (int)gen_len; step_++) {
        // Loop body produces Nth token by embedding && encoding token (N-1)
        // if necessary.
        const bool fill_caches_only = continue_gen && (step_ < max_context_len);
        const int  src_indir_idx    = (step_ - step_start) % 2;
        const int  tgt_indir_idx    = 1 - src_indir_idx;

        const size_t iteration_num = batch_size / local_batch_size;
        *generation_should_stop_   = !fill_caches_only;

        PUSH_RANGE(fmtstr("token_%d", step_ - step_start));

        printf("[%s:%d-pid:%d]: <-- ite loop --> token_%d <---- step_:%d---->, 【iteration_num=%d】 fill_caches_only=%d, src_indir_idx=%d, tgt_indir_idx=%d\n", 
            __FUNCTION__, __LINE__, getpid(), 
            step_ - step_start,
            step_,
            iteration_num,
            fill_caches_only,
            src_indir_idx,
            tgt_indir_idx);
        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset               = ite * local_batch_size * beam_width;
            const int hidden_units_offset     = id_offset * hidden_units_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            printf("[%s:%d-pid:%d]: <----step_:%d/ite:%d----> id_offset=%d hidden_units_offset=%d vocab_size_units_offset=%d\n", 
                __FUNCTION__, __LINE__, getpid(), 
                step_, ite, id_offset, hidden_units_offset, vocab_size_units_offset);

            // Rank 0~N-1 needs to update the buffer by the results of last rank when the pipeline parallelism is
            // enabled (pipeline_para_.world_size_ > 1). And if step_ == step_start, then this is the first step and
            // these buffers are initialized by context directly.
            if (step_ != step_start && pipeline_para_.rank_ != pipeline_para_.world_size_ - 1
                && pipeline_para_.world_size_ > 1) {
                ftNcclGroupStart();
                // receive updated sequence_length_ from last rank
                ftNcclRecv(sequence_lengths_ + id_offset,
                           local_batch_size * beam_width,
                           pipeline_para_.world_size_ - 1,
                           pipeline_para_,
                           stream_);

                // receive updated generation_should_stop_ from last rank
                if (ite == 0) {
                    ftNcclRecv(generation_should_stop_, 1, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);
                }

                // receive updated cache_indirections from last rank
                if (beam_width > 1) {
                    ftNcclRecv(cache_indirections_[tgt_indir_idx] + id_offset * memory_len,
                               local_batch_size * beam_width * memory_len,
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                // for ids of next step, only first rank needs to receive updated ids
                if (pipeline_para_.rank_ == 0) {
                    ftNcclRecv(output_ids_buf_ + (step_ - 1) * batch_size * beam_width + id_offset,
                               local_batch_size * beam_width,
                               pipeline_para_.world_size_ - 1,
                               pipeline_para_,
                               stream_);
                }

                ftNcclGroupEnd();
                // throw errors when detected
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                sync_check_cuda_error();

                if (ite == 0 && *generation_should_stop_) {
                    break;
                }
            }

            if ((max_input_length <= 1) || (step_ > step_start) || continue_gen) {
                if (pipeline_para_.rank_ == 0) {
                    printf("[%s:%d-pid:%d]: <invokeEmbeddingLookupPosEncodingPadCount> <----step_:%d/ite:%d----> local_token_num%d hidden_units:%d step:%d token_num%d ite%d\n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite,
                        local_batch_size * beam_width,
                        hidden_units_,
                        step_ - 1,
                        batch_size * beam_width,
                        0);
                        
                    invokeEmbeddingLookupPosEncodingPadCount(decoder_input_buf_ + hidden_units_offset,
                                                             gpt_weights->pre_decoder_embedding_table,
                                                             gpt_weights->position_encoding_table,
                                                             output_ids_buf_ + id_offset,
                                                             tiled_total_padding_count_ + id_offset,
                                                             local_batch_size * beam_width,
                                                             hidden_units_,
                                                             (T)(1.0f),
                                                             step_ - 1,
                                                             batch_size * beam_width,
                                                             0,
                                                             stream_);
                    sync_check_cuda_error();

                    if (gpt_variant_params_.has_pre_decoder_layernorm) {
                        printf("[%s:%d-pid:%d]: <invokeGeneralLayerNorm> <----step_:%d/ite:%d----> \n", 
                            __FUNCTION__, __LINE__, getpid(),
                            step_, ite);
                        invokeGeneralLayerNorm(decoder_normed_input_buf_ + hidden_units_offset,
                                               decoder_input_buf_ + hidden_units_offset,
                                               gpt_weights->pre_decoder_layernorm.gamma,//
                                               gpt_weights->pre_decoder_layernorm.beta,//
                                               layernorm_eps_,
                                               batch_size * beam_width,
                                               hidden_units_,
                                               (float*)nullptr,
                                               0,
                                               stream_);
                    }
                    sync_check_cuda_error();
                }

                std::unordered_map<std::string, Tensor> decoder_input_tensors(
                    {{"decoder_input",
                      Tensor(MEMORY_GPU,
                             data_type,
                             {local_batch_size * beam_width, hidden_units_},
                             gpt_variant_params_.has_pre_decoder_layernorm ?
                                 decoder_normed_input_buf_ + hidden_units_offset :
                                 decoder_input_buf_ + hidden_units_offset)},
                     {"finished",
                      Tensor(MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset)},
                     {"input_lengths",
                      Tensor(MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths_ + id_offset)},
                     {"total_padding_tokens",
                      Tensor(MEMORY_GPU,
                             TYPE_INT32,
                             {local_batch_size * beam_width},
                             tiled_total_padding_count_ + id_offset)},
                     {"max_input_length", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &max_context_len)},
                     {"step", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &step_)},
                     {"ite", Tensor(MEMORY_CPU, TYPE_INT32, {1}, &ite)},
                     {"masked_tokens",
                      Tensor(MEMORY_GPU,
                             TYPE_BOOL,
                             {local_batch_size * beam_width, memory_len},
                             masked_tokens_ + id_offset * memory_len)}});
                if (beam_width > 1) {
                    decoder_input_tensors.insert({"cache_indirection",
                                                  Tensor(MEMORY_GPU,
                                                         TYPE_INT32,
                                                         {local_batch_size, beam_width, memory_len},
                                                         cache_indirections_[src_indir_idx] + id_offset * memory_len)});
                }

                if (gpt_variant_params_.use_attention_linear_bias) {
                    decoder_input_tensors.insert({"linear_bias_slopes",
                                                  Tensor(MEMORY_GPU,
                                                         data_type,
                                                         {local_head_num_},
                                                         linear_bias_slopes_ + local_head_num_ * tensor_para_.rank_)});
                }

                std::unordered_map<std::string, Tensor> decoder_output_tensors(
                    {{"decoder_output",
                      Tensor(MEMORY_GPU,
                             data_type,
                             {local_batch_size * beam_width, hidden_units_},
                             decoder_output_buf_ + hidden_units_offset)},
                     {"key_cache", Tensor(MEMORY_GPU, data_type, self_k_cache_shape, key_cache_)},
                     {"value_cache", Tensor(MEMORY_GPU, data_type, self_v_cache_shape, value_cache_)}});



                {
                    int out_size = batch_size;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), sequence_lengths_, out_size);
                    printf("-----------------sequence_lengths_ [gpt_decoder_->forward pre] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    } 
                    printf("\n");
                }

                //除第一个decoder外的所有decoder （gpt_decoder_）
                printf("[%s:%d-pid:%d]: <gpt_decoder_->forward> <----step_:%d/ite:%d----> \n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                gpt_decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);

                {
                    int out_size = 40;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), output_ids_buf_, out_size);
                    printf("gpt_decoder_->forward param.step_ids[output_ids_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    }
                    printf("\n");
                }
            }

            if (!fill_caches_only && pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                // OPT
                PUSH_RANGE("Token Final Layer Norm");
                T* decoder_output_final_buf =
                    gpt_variant_params_.has_post_decoder_layernorm ? normed_decoder_output_buf_ : decoder_output_buf_;
                if (gpt_variant_params_.has_post_decoder_layernorm) {
                    printf("[%s:%d-pid:%d]:  <invokeGeneralLayerNorm> <----step_:%d/ite:%d----> [Token Final Layer Norm] layernorm_eps:%f m:%d n:%d\n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite,
                        layernorm_eps_, local_batch_size * beam_width, hidden_units_);
                    invokeGeneralLayerNorm(normed_decoder_output_buf_ + hidden_units_offset,//out
                                           decoder_output_buf_ + hidden_units_offset,//in
                                           gpt_weights->post_decoder_layernorm.gamma,
                                           gpt_weights->post_decoder_layernorm.beta,
                                           layernorm_eps_,
                                           local_batch_size * beam_width,
                                           hidden_units_,
                                           (float*)nullptr,
                                           0,
                                           stream_);
                }
                sync_check_cuda_error();
                POP_RANGE;

                if (tensor_para_.world_size_ == 1) {
                    float alpha = 1.0f;
                    float beta  = 0.0f;
                    PUSH_RANGE("logits gemm");
                    printf("[%s:%d-pid:%d]: <cublas_wrapper_->Gemm> [logits gemm] <----step_:%d/ite:%d----> m=%d n=%d k=%d\n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite,
                        local_batch_size * beam_width,
                        vocab_size_padded_,
                        hidden_units_);
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,//alpha  1.0
                                          padded_embedding_kernel_ptr_,//A
                                          gemm_data_type,//a type
                                          hidden_units_,   //lda                                // k
                                          decoder_output_final_buf + hidden_units_offset,  // OPT: no final layer norm B
                                          gemm_data_type,//b type
                                          hidden_units_,  // k ldb
                                          &beta,//beta 0.0
                                          logits_buf_ + vocab_size_units_offset,//C
                                          CUDA_R_32F,//c type 
                                          vocab_size_padded_, /* n */ //ldc
                                          CUDA_R_32F,//computeType
                                          cublasGemmAlgo_t(-1)); //algo
                    POP_RANGE;
                }
                else {
                    FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    float     alpha            = 1.0f;
                    float     beta             = 0.0f;
                    PUSH_RANGE("logits gemm");
                    printf("[%s:%d-pid:%d]: <cublas_wrapper_->Gemm>: [logits gemm] <----step_:%d/ite:%d---->\n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite);
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                          gemm_data_type,
                                          hidden_units_,                                   // k
                                          decoder_output_final_buf + hidden_units_offset,  // OPT: no final layer norm
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                                          CUDA_R_32F,
                                          local_vocab_size, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                    POP_RANGE;
                    PUSH_RANGE("logits all reduce sum");
                    printf("[%s:%d-pid:%d]: <ftNcclAllGather> [logits all reduce sum] <----step_:%d/ite:%d---->\n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite);
                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                    printf("[%s:%d-pid:%d]: <invokeTransposeAxis01> <----step_:%d/ite:%d----> \n", 
                        __FUNCTION__, __LINE__, getpid(),
                        step_, ite);
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width,
                                          local_vocab_size,
                                          stream_);
                    POP_RANGE;
                }

                int  tmp_local_batch_size       = local_batch_size;
                bool is_initialize_random_table = step_ == max_context_len;

                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step_}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_context_len}},
                    {"sequence_limit_length", Tensor{MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len_}},//1
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                    {"input_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, memory_len},
                            cache_indirections_[src_indir_idx] + id_offset * memory_len}},
                    {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
                    {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

                for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
                    if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }

                // common outputs
                bool                                    subbatch_should_stop = false;
                std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
                    {"output_ids", Tensor{MEMORY_GPU, TYPE_INT32, {gen_len, batch_size, beam_width}, output_ids_buf_}},
                    {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_}},
                    // cum_log_probs is necessary for beam search, while it is optional for sampling.
                    {"cum_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {batch_size * beam_width},
                            ((beam_width > 1) || (output_tensors->count("cum_log_probs") > 0)) ? cum_log_probs_ :
                                                                                                 nullptr}},
                    {"output_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {gen_len, batch_size, beam_width},
                            output_tensors->count("output_log_probs") > 0
                                    && output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ :
                                nullptr}},
                    {"parent_ids", Tensor{MEMORY_GPU, TYPE_INT32, {gen_len, batch_size, beam_width}, parent_ids_buf_}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths_}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, memory_len},
                            cache_indirections_[tgt_indir_idx] + id_offset * memory_len}},
                    {"should_stop", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &subbatch_should_stop}}};
                for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }

                PUSH_RANGE("result sampling and stop check");
                // forward:topk  （dynamic_decode_layer_）
                printf("[%s:%d-pid:%d]: <dynamic_decode_layer_->forward>: [result sampling and stop check] <----step_:%d/ite:%d---->\n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                *generation_should_stop_ &= subbatch_should_stop;

                {
                    int out_size = 40;
                    std::vector<int> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), output_ids_buf_, out_size);
                    printf("dynamic_decode_layer_->forwar param.step_ids[output_ids_buf_] h_buf:\n");
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", h_buf[k]);
                    }
                    printf("\n");
                }

                POP_RANGE;
            }

            PUSH_RANGE("result communication");
            // send results to other rank
            if (fill_caches_only) {
                printf("[%s:%d-pid:%d]:<invokePlusScalar> [result communication] <----step_:%d/ite:%d---->\n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                invokePlusScalar(sequence_lengths_, 1, batch_size * beam_width, stream_);
            }

            // When pipeline parallelism is enabled (pipeline_para_.world_size_ > 1), last rank needs to send updates
            // to other ranks.
            printf("[%s:%d-pid:%d]: <----step_:%d/ite:%d----> gen_len=%d, step_=%d, world_size_=%d, rank_=%d \n", 
                __FUNCTION__, __LINE__, getpid(), 
                step_, ite,
                gen_len, step_, pipeline_para_.world_size_, pipeline_para_.rank_);
            if (step_ < gen_len - 1 && pipeline_para_.world_size_ > 1
                && pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                printf("[%s:%d-pid:%d]: <ftNcclGroupStart> <----step_:%d/ite:%d----> \n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                ftNcclGroupStart();
                for (int i = 0; i < pipeline_para_.world_size_ - 1; i++) {
                    // send updated sequence_length_ to other rank
                    ftNcclSend(
                        sequence_lengths_ + id_offset, local_batch_size * beam_width, i, pipeline_para_, stream_);

                    // send updated generation_should_stop_
                    if (ite == 0) {
                        ftNcclSend(generation_should_stop_, 1, i, pipeline_para_, stream_);
                    }

                    // send updated cache_indirections
                    if (beam_width > 1) {
                        ftNcclSend(cache_indirections_[tgt_indir_idx] + id_offset * memory_len,
                                   local_batch_size * beam_width * memory_len,
                                   i,
                                   pipeline_para_,
                                   stream_);
                    }
                }

                printf("[%s:%d-pid:%d]: <ftNcclSend> <----step_:%d/ite:%d----> \n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                // for ids of next step, only need to send updated ids to first rank
                ftNcclSend(output_ids_buf_ + step_ * batch_size * beam_width + id_offset,
                           local_batch_size * beam_width,
                           0,
                           pipeline_para_,
                           stream_);

                printf("[%s:%d-pid:%d]:<ftNcclGroupEnd> <----step_:%d/ite:%d----> \n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                ftNcclGroupEnd();
                // throw errors when detected
                printf("[%s:%d-pid:%d]:  <ftNcclStreamSynchronize> <----step_:%d/ite:%d---->\n", 
                    __FUNCTION__, __LINE__, getpid(),
                    step_, ite);
                ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
                sync_check_cuda_error();
            }
            POP_RANGE;
        }//for (uint ite = 0; ite < iteration_num; ++ite)循环结束

        if (token_generated_cb_ && step_ + 1 < (int)gen_len) {
            printf("[%s:%d-pid:%d]: <setOutputTensors> <----step_:%d---->\n", __FUNCTION__, __LINE__, getpid(),step_);
            setOutputTensors(
                output_tensors, input_tensors, gen_len, session_len, max_context_len, max_input_without_prompt_length);
            printf("[%s:%d-pid:%d]: <sendTensorsToFirstPipelineNode> <----step_:%d---->\n", __FUNCTION__, __LINE__, getpid(),step_);
            sendTensorsToFirstPipelineNode(output_tensors, input_tensors);

            if (pipeline_para_.rank_ == 0 && tensor_para_.rank_ == 0) {
                token_generated_cb_(output_tensors, token_generated_ctx_);
            }
        }

        if (step_ == initial_step + max_input_length) {
            /* We have just finished processing input: update the padding count:
             * total_padding_count += (max_input_length - input_lengths) */
            printf("[%s:%d-pid:%d]: <invokeUpdatePaddingCount> <----step_:%d---->\n", __FUNCTION__, __LINE__, getpid(),step_);
            invokeUpdatePaddingCount(tiled_total_padding_count_,
                                     input_tensors->at("input_lengths").getPtr<int>(),
                                     max_input_length,
                                     batch_size,
                                     beam_width,
                                     stream_);
        }
        POP_RANGE;
    }//for (step_ = step_start; step_ < (int)gen_len; step_++)循环结束
    PUSH_RANGE("communicate tensors");
    printf("[%s:%d-pid:%d]: <setOutputTensors> [communicate tensors]\n", __FUNCTION__, __LINE__, getpid());
    setOutputTensors(
        output_tensors, input_tensors, gen_len, session_len, max_context_len, max_input_without_prompt_length);
    printf("[%s:%d-pid:%d]: <sendTensorsToFirstPipelineNode> []\n", __FUNCTION__, __LINE__, getpid());
    sendTensorsToFirstPipelineNode(output_tensors, input_tensors);
    POP_RANGE;
}

template<typename T>
void ParallelGpt<T>::sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                                    const std::unordered_map<std::string, Tensor>* input_tensors)
{
    printf("pipeline_para_.world_size_:%d\n",pipeline_para_.world_size_);
    if (pipeline_para_.world_size_ == 1) {
        {
            int out_size = output_tensors->at("output_ids").size();
            std::vector<int> h_buf(out_size);
            cudaD2Hcpy(h_buf.data(), output_tensors->at("output_ids").getPtr<int>(), out_size);
            printf("h_buf:\n");
            for(int k=0; k<out_size; k++) {
                printf("%d ", h_buf[k]);
            }
            printf("\n");
        }
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);

        int out_size = output_tensors->at("output_ids").size();
        std::vector<int> h_buf(out_size);
        cudaD2Hcpy(h_buf.data(), output_tensors->at("output_ids").getPtr<int>(), out_size);
        printf("h_buf:\n");
        for(int k=0; k<out_size; k++) {
            printf("%d ", h_buf[k]);
        }
        printf("\n");

        return;
    }

    const auto pp_rank = pipeline_para_.rank_;

    ftNcclGroupStart();
    for (auto const& it : *output_tensors) {
        if (it.second.data == nullptr) {
            continue;
        }

        if (pp_rank == pipeline_para_.world_size_ - 1) {
            ftNcclSend(it.second.getPtr<char>(), it.second.sizeBytes(), 0, pipeline_para_, stream_);
        }
        else if (pp_rank == 0) {
            ftNcclRecv(it.second.getPtr<char>(),
                       it.second.sizeBytes(),
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();
    // throw errors when detected
     (tensor_para_, pipeline_para_, stream_);
}

template<typename T>
void ParallelGpt<T>::setOutputTensors(std::unordered_map<std::string, Tensor>*       output_tensors,
                                      const std::unordered_map<std::string, Tensor>* input_tensors,
                                      const size_t                                   gen_len,
                                      const size_t                                   session_len,
                                      const size_t                                   max_context_len,
                                      const size_t                                   max_input_without_prompt_length)
{
    if (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
        return;
    }

    const size_t batch_size       = output_tensors->at("output_ids").shape[0];
    const size_t beam_width       = output_tensors->at("output_ids").shape[1];
    int*         sequence_lengths = output_tensors->at("sequence_length").getPtr<int>();
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;

    cudaAutoCpy(sequence_lengths, sequence_lengths_, output_tensors->at("sequence_length").size(), stream_);
    printf("output_tensors shape:{%d,%d} size:%d, sequence_length size:%d input_ids:{%d,%d}\n",
        output_tensors->at("output_ids").shape[0],//8
        output_tensors->at("output_ids").shape[1],
        output_tensors->at("output_ids").size(),
        output_tensors->at("sequence_length").size(),
        input_tensors->at("input_ids").shape[0],input_tensors->at("input_ids").shape[1]
        );
    
    //cudaDeviceSynchronize();
    int out_size = output_tensors->at("output_ids").size();
    std::vector<int> h_buf(out_size);
    cudaD2Hcpy(h_buf.data(), output_tensors->at("output_ids").getPtr<int>(), out_size);
    printf("h_buf:\n");
    for(int k=0; k<out_size; k++) {
        printf("%d ", h_buf[k]);
    }
    printf("\n");

    int sequence_out_size = output_tensors->at("sequence_length").size();
    std::vector<int> h_sequence_buf(sequence_out_size);
    cudaD2Hcpy(h_sequence_buf.data(), sequence_lengths, sequence_out_size);
    printf("h_sequence_buf:\n");
    for(int k=0; k<sequence_out_size; k++) {
        printf("%d ", h_sequence_buf[k]);
    }
    printf("\n");

    if (input_tensors->at("input_ids").shape[1] == 0) {
        // TODO: D2D sequence_lenghts
        if (beam_width > 1) {
            // For beam search, do gather_tree
            // take output_parent_ids as inter buffer
            invokeGatherTree(transposed_output_ids_buf_,
                             sequence_lengths_,
                             session_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batch_size * beam_width,
                             parent_ids_buf_ + batch_size * beam_width,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  transposed_output_ids_buf_,
                                  gen_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        else {
            // For sampling, only copy the results to output_tensor
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  output_ids_buf_ + batch_size * beam_width,
                                  gen_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
    }
    else {
        // For sampling, it is equivalent to all parent ids are 0.
        gatherTreeParam param;
        param.beams = transposed_output_ids_buf_;
        // Remove prompt length if possible
        param.max_sequence_lengths = sequence_lengths;
        // add sequence_length 1 here because the sequence_length of time step t is t - 1
        param.max_sequence_length_final_step = 1;
        // response input lengths (used to slice the ids during postprocessing)
        param.response_input_lengths = output_tensors->count("response_input_lengths") ?
                                           output_tensors->at("response_input_lengths").getPtr<int>() :
                                           nullptr;
        param.max_time               = gen_len;
        param.batch_size             = batch_size;
        param.beam_width             = beam_width;
        param.step_ids               = output_ids_buf_;
        param.parent_ids             = beam_width == 1 ? nullptr : parent_ids_buf_;
        param.end_tokens             = end_ids_buf_;
        param.max_input_length       = max_context_len;
        param.prefix_soft_prompt_lengths =
            has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_lengths").getPtr<int>() : nullptr;
        param.input_lengths                   = tiled_input_lengths_buf_;
        param.p_prompt_tuning_prompt_lengths  = has_p_prompt_tuning_ ? tiled_prompt_lengths_buf_ : nullptr;
        param.max_input_without_prompt_length = max_input_without_prompt_length;
        param.max_prefix_soft_prompt_length   = max_prefix_soft_prompt_length;
        param.stream                          = stream_;
        param.output_ids                      = output_tensors->at("output_ids").getPtr<int>();

    /*
    int*       beams                          = nullptr;
    int*       max_sequence_lengths           = nullptr;
    int        max_sequence_length_final_step = 0;
    const int* input_lengths                  = nullptr;
    // response input lengths (used to slice the ids during postprocessing)
    int*       response_input_lengths     = nullptr;
    int        max_time                   = 0;
    int        batch_size                 = 0;
    int        beam_width                 = 0;
    const int* step_ids                   = nullptr;
    const int* parent_ids                 = nullptr;
    const int* end_tokens                 = nullptr;
    int        max_input_length           = 0;
    const int* prefix_soft_prompt_lengths = nullptr;
    // p_prompt_tuning prompt leangths, used to remove prompts during post-processing
    const int* p_prompt_tuning_prompt_lengths  = nullptr;
    int        max_input_without_prompt_length = 0;
    // prefix soft prompt
    int          max_prefix_soft_prompt_length = 0;
    int*         output_ids                    = nullptr;
    cudaStream_t stream;
    */

        printf("param: %d %d %d %d %d %d %d response_input_lengths=%d\n",
            param.max_sequence_length_final_step,//1
            param.max_time,//40
            param.batch_size,//1
            param.beam_width,//1
            param.max_input_length,//8
            param.max_input_without_prompt_length,//8
            param.max_prefix_soft_prompt_length,//0
            output_tensors->count("response_input_lengths"));

        {
            //param.step_ids output_ids_buf_
            int out_size = param.max_time;
            std::vector<int> h_buf(out_size);
            cudaD2Hcpy(h_buf.data(), param.step_ids, out_size);
            printf("param.step_ids[output_ids_buf_] h_buf:\n");
            for(int k=0; k<out_size; k++) {
                printf("%d ", h_buf[k]);
            }
            printf("\n");
        }

        {
            //transposed_output_ids_buf_
            int out_size = param.max_time;
            std::vector<int> h_buf(out_size);
            cudaD2Hcpy(h_buf.data(), transposed_output_ids_buf_, out_size);
            printf("transposed_output_ids_buf_ h_buf:\n");
            for(int k=0; k<out_size; k++) {
                printf("%d ", h_buf[k]);
            }
            printf("\n");
        }

        {
            int out_size = output_tensors->at("output_ids").size();
            std::vector<int> h_buf(out_size);
            cudaD2Hcpy(h_buf.data(), output_tensors->at("output_ids").getPtr<int>(), out_size);
            printf("h_buf:\n");
            for(int k=0; k<out_size; k++) {
                printf("%d ", h_buf[k]);
            }
            printf("\n");
        }

        // NOTE: need to remove all prompt virtual tokens
        invokeGatherTree(param);

        {
            int out_size = output_tensors->at("output_ids").size();
            std::vector<int> h_buf(out_size);
            cudaD2Hcpy(h_buf.data(), output_tensors->at("output_ids").getPtr<int>(), out_size);
            printf("h_buf:\n");
            for(int k=0; k<out_size; k++) {
                printf("%d ", h_buf[k]);
            }
            printf("\n");
        }
        sync_check_cuda_error();
    }

    printf("has_p_prompt_tuning_[%d], output_log_probs:%d, cum_log_probs:%d  is_finished:%d\n",
        has_p_prompt_tuning_,
        output_tensors->count("output_log_probs"),
        output_tensors->count("cum_log_probs"),
        output_tensors->count("is_finished"));

    // remove p_prompt virtual tokens and update output tensors shape
    if (has_p_prompt_tuning_) {  // remove p_prompt virtual tokens and update output tensors shape
        output_tensors->at("output_ids").updateShape(2, gen_len - (max_context_len - max_input_without_prompt_length));
    }//不执行

    if ((output_tensors->count("output_log_probs") > 0 && output_tensors->at("output_log_probs").data != nullptr)) {
        invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                              output_log_probs_buf_,
                              input_tensors->at("output_seq_len").max<uint32_t>() - max_context_len,
                              batch_size * beam_width,
                              1,
                              stream_);
    }//不执行
    // Return the cumulative log probability if requested.
    if (output_tensors->count("cum_log_probs") > 0) {
        Tensor cum_log_probs = output_tensors->at("cum_log_probs");
        FT_CHECK_WITH_INFO(cum_log_probs.size() == batch_size * beam_width,
                           "The shape of cum_log_probs does not match with batch_size x beam_width.");
        cudaAutoCpy(cum_log_probs.getPtr<float>(), cum_log_probs_, cum_log_probs.size(), stream_);
    }//不执行

    if (output_tensors->count("is_finished")) {
        cudaD2Dcpy(
            output_tensors->at("is_finished").getPtr<bool>(), finished_buf_, output_tensors->at("is_finished").size());
    }//不执行
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelRank()
{
    return pipeline_para_.rank_;
}

template<typename T>
size_t ParallelGpt<T>::getPipelineParallelSize()
{
    return pipeline_para_.world_size_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelRank()
{
    return tensor_para_.rank_;
}

template<typename T>
size_t ParallelGpt<T>::getTensorParallelSize()
{
    return tensor_para_.world_size_;
}

template<typename T>
size_t ParallelGpt<T>::getHiddenUnits()
{
    return hidden_units_;
}

template<typename T>
bool* ParallelGpt<T>::getFinishBuffer()
{
    return finished_buf_;
}

template<typename T>
size_t ParallelGpt<T>::getStep()
{
    return step_;
}

template class ParallelGpt<float>;
template class ParallelGpt<half>;
#ifdef ENABLE_BF16
template class ParallelGpt<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
