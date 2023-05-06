/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h"
#include "src/fastertransformer/kernels/unfused_attention_kernels.h"
#include "src/fastertransformer/utils/nvtx_utils.h"


#include <unistd.h>
#include <sys/syscall.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace fastertransformer {

template<typename T>
void GptContextAttentionLayer<T>::forward(TensorMap*                output_tensors,
                                          TensorMap*                input_tensors,
                                          const AttentionWeight<T>* attention_weights)
{
    // input_tensors:
    //      input_query [token_num, hidden_dimension]
    //      attention_mask [batch_size, 1, seq_len, seq_len + max_prompt_length]
    //      attention_type [1]
    //      is_final_layer [1], bool on cpu
    //      layer_id [1], int on cpu
    //      padding_offset, int, [token_num] (optional)
    //      cu_seqlens, int, [batch_size] (optional)
    //      d_prefix_prompt_batch [global_batch_size], (optional)
    //          each element contains ptr with buffer shape[2, local_head_num_, prompt_length, size_per_head]
    //      d_prefix_prompt_lengths [batch_size], int (optional)
    //      linear_bias_slopes [head_num] (optional)

    // output_tensors:
    //      hidden_features [token_num, hidden_dimension]
    //      key_cache [batch, local_head_num, size_per_head // x, max_seq_len, x]
    //      value_cache [batch, local_head_num, max_seq_len, size_per_head]
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    FT_CHECK(output_tensors->at("key_cache").shape.size() == 5);
    FT_CHECK(output_tensors->at("value_cache").shape.size() == 4
             || output_tensors->at("value_cache").shape.size() == 3);
    const int request_batch_size = input_tensors->at("attention_mask").shape[0];
    const int request_seq_len    = input_tensors->at("attention_mask").shape[2];
    const int max_prompt_length =
        input_tensors->at("attention_mask").shape[3] - input_tensors->at("attention_mask").shape[2];
    const int  layer_id                = input_tensors->getVal<int>("layer_id");
    const T**  d_prefix_prompt_batch   = input_tensors->getPtr<const T*>("d_prefix_prompt_batch", nullptr);
    const int* d_prefix_prompt_lengths = input_tensors->getPtr<int>("d_prefix_prompt_lengths", nullptr);
    const int* padding_offset          = input_tensors->getPtr<int>("padding_offset", nullptr);
    int*       cu_seqlens              = input_tensors->getPtr<int>("cu_seqlens", nullptr);
    T*         linear_bias_slopes      = input_tensors->getPtr<T>("linear_bias_slopes", nullptr);
    /* float*     attention_query_dynamic_scale = input_tensors->getPtr<float>("attention_query_dynamic_scale",
     * nullptr); */

    T* attention_out   = output_tensors->at("hidden_features").getPtr<T>();
    T* attention_input = input_tensors->at("input_query").getPtr<T>();
    T* attention_mask  = input_tensors->at("attention_mask").getPtr<T>();

    const AttentionType attention_type = input_tensors->getVal<AttentionType>("attention_type");
    FT_CHECK_WITH_INFO(attention_type != AttentionType::FUSED_PADDED_MHA,
                       "Gpt Context FUSED_PADDED_MHA is not supported !");

    PUSH_RANGE("attention buffer alloc");
    allocateBuffer(request_batch_size, request_seq_len + max_prompt_length, attention_type != AttentionType::FUSED_MHA);
    //batch_size=request_batch_size=1 seq_len=request_seq_len + max_prompt_length=8
    POP_RANGE;
    sync_check_cuda_error();

    const bool is_final = input_tensors->at("is_final_layer").getVal<bool>();

    const int m = input_tensors->at("input_query").shape[0];

    PUSH_RANGE("qkv_gemm");

#ifdef SPARSITY_ENABLED
    const int m_padded   = 8 * div_up(m, 8);
    bool      use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, 3 * local_hidden_units_, m_padded, hidden_units_);
#else
    constexpr bool use_sparse = false;
#endif

    printf("[%s:%d-pid:%d-%s]: <......begin.....> use_sparse=%d int8_mode_=%d attention_mask.size=%d\n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__,
        use_sparse, int8_mode_,input_tensors->at("attention_mask").size());

    if (use_sparse) {
#ifdef SPARSITY_ENABLED
        printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->SpGemm>  \n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__);
        cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                3 * local_hidden_units_,
                                m_padded,
                                hidden_units_,
                                attention_weights->query_weight.sp_kernel,
                                attention_input,
                                qkv_buf_);
#endif
    }
    else if (int8_mode_ == 1) {
        FT_CHECK(weight_only_int8_fc_runner_.get() != NULL && attention_weights->query_weight.int8_kernel != NULL
                 && attention_weights->query_weight.weight_only_quant_scale != NULL);

        printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->SpGemm>  \n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__);
        weight_only_int8_fc_runner_->gemm(attention_input,
                                          reinterpret_cast<const uint8_t*>(attention_weights->query_weight.int8_kernel),
                                          attention_weights->query_weight.weight_only_quant_scale,
                                          qkv_buf_,
                                          m,
                                          3 * local_hidden_units_,
                                          hidden_units_,
                                          mixed_gemm_workspace_,
                                          mixed_gemm_ws_bytes_,
                                          stream_);
    }
    else if (int8_mode_ == 2) {
        printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->Int8Gemm>  \n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__);
        cublas_wrapper_->Int8Gemm(3 * local_hidden_units_,
                                  m,
                                  hidden_units_,
                                  attention_weights->query_weight.int8_kernel,
                                  hidden_units_,
                                  input_tensors->at("input_query").getPtr<int8_t>(),
                                  hidden_units_,
                                  reinterpret_cast<int8_t*>(qkv_buf_),
                                  3 * local_hidden_units_,
                                  attention_weights->query_weight.scale_inter,
                                  true);
    }
    else {
        printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->Gemm> m%dn%dk%d \n", //<cublas_wrapper_->Gemm> m8n3072k1024 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__,
            m,//8
            3 * local_hidden_units_,//3072=3*1024
            hidden_units_);//1024
        cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * local_hidden_units_,  // n 3*1024=3072
                              m,//8
                              hidden_units_,  // k 1024
                              attention_weights->query_weight.kernel,
                              3 * local_hidden_units_,  // n
                              attention_input,
                              hidden_units_,  // k
                              qkv_buf_,
                              3 * local_hidden_units_ /* n */);
    }

    sync_check_cuda_error();

    // IDEA: append prefix prompt key value here
    PrefixPromptBatchWeightsParam<T> param{d_prefix_prompt_batch,
                                           d_prefix_prompt_lengths,
                                           max_prompt_length,
                                           (size_t)layer_id * 2 * local_head_num_ * size_per_head_};

    if (padding_offset != nullptr) {
        // q_buf_2_, k_buf_2_ and v_buf_2_ are continuous
        cudaMemsetAsync(
            q_buf_2_, 0, request_batch_size * request_seq_len * 3 * local_hidden_units_ * sizeof(T), stream_);
    }

    printf("[%s:%d-pid:%d-%s]: <invokeAddFusedQKVBiasTranspose>  padding_offset=%p batch:%d seq_len:%d token:%d, %d %d, rotary_embedding_dim_:%d %d\n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__,
        padding_offset,
        request_batch_size,// batch:1
        request_seq_len,//8
        m,//
        local_head_num_,//16
        size_per_head_,//64
        rotary_embedding_dim_,
        neox_rotary_style_);
    invokeAddFusedQKVBiasTranspose(q_buf_2_,//8192  batch_size:1 seq_len:8 local_hidden_units_:1024
                                   k_buf_2_,//8192  batch_size:1 seq_len:8 local_hidden_units_:1024
                                   v_buf_2_,//8192  batch_size:1 seq_len:8 local_hidden_units_:1024
                                   param,  // prefix prompt
                                   qkv_buf_,//49152  24576=3*1(batch_size)*8(seq_len)*1024(local_hidden_units_)
                                   attention_weights->query_weight.bias,
                                   padding_offset,
                                   request_batch_size,// batch:1
                                   request_seq_len,//seq:8
                                   m,//token:8
                                   local_head_num_,//16
                                   size_per_head_,//64
                                   rotary_embedding_dim_,//0
                                   neox_rotary_style_,//0
                                   attention_weights->query_weight.scale_out,
                                   int8_mode_,
                                   stream_);
    sync_check_cuda_error();

    const int max_seq_len = (int)(output_tensors->at("key_cache").shape[3]);  // max output seq length
    // Use batch major
    // put k/v_buf from shape [B, H, PL + L, Dh] :            //[batch, head_num, seq_len, size_per_head]
    // to cache [B, H, Dh/x, PL + L, x]  and [B, H, PL + L, Dh/x, x], PL denotes prompt length
    printf("[%s:%d-pid:%d-%s]: <invokeTranspose4dBatchMajor>  \n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__);
    invokeTranspose4dBatchMajor(output_tensors->getPtr<T>("key_cache"),
                                output_tensors->getPtr<T>("value_cache"),
                                k_buf_2_,
                                v_buf_2_,
                                request_batch_size,
                                max_prompt_length + request_seq_len,  // max input length + prefix prompt length
                                max_seq_len,
                                size_per_head_,
                                local_head_num_,
                                stream_);
    // IDEA : after this, k_cache = (batch_size, num_heads, Dh/x, prefix_prompt_len + L, x)
    // k_cache = (batch_size, num_heads, prefix_prompt_len + L, Dh)
    sync_check_cuda_error();

    // TODO: fmha kernels doesn't support different seq lengths of q and kv
    if (attention_type == AttentionType::FUSED_MHA) {
        printf("[%s:%d-pid:%d-%s]: <dispatcher_fp16->setup_causal_masked_fmha>  \n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__);
        dispatcher_fp16->setup_causal_masked_fmha(request_seq_len, request_batch_size);
        dispatcher_fp16->run_causal_masked_fmha(qkv_buf_, cu_seqlens, qkv_buf_3_, true, stream_);
    }
    // NOTE: qkv buffer shape (batch_size, num_heads,L or prompt_len + L, Dh)

    POP_RANGE;
    if (is_final == false) {
        const cudaDataType_t gemm_data_type      = getCudaDataType<T>();
        const int            attention_seq_len_1 = request_seq_len;                      // q length
        const int            attention_seq_len_2 = max_prompt_length + request_seq_len;  // kv length
        const T              qk_scale            = static_cast<T>(1.0f / sqrtf(size_per_head_ * 1.0f));
        printf("[%s:%d-pid:%d-%s]: <>  q[%d] k/v[%d] max_prompt_length=%d request_seq_len=%d\n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__,
            attention_seq_len_1,
            attention_seq_len_2,
            max_prompt_length, request_seq_len);
        if (attention_type != AttentionType::FUSED_MHA) {
            if (is_qk_buf_float_ == true && gemm_data_type != CUDA_R_32F) {
                PUSH_RANGE("Q*K batch gemm");
                printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->stridedBatchedGemm> [Q*K batch gemm] m%dn%dk%d \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    attention_seq_len_1,attention_seq_len_2,size_per_head_);
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2,  // n 8 q
                                                    attention_seq_len_1,  // m 8 k/v
                                                    size_per_head_,       // k 64
                                                    1.0f,
                                                    k_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_2 * size_per_head_,  // n * k
                                                    q_buf_2_,
                                                    gemm_data_type,
                                                    size_per_head_,                        // k
                                                    attention_seq_len_1 * size_per_head_,  // m * k
                                                    0.0f,
                                                    qk_buf_float_,
                                                    CUDA_R_32F,
                                                    attention_seq_len_2,  // n
                                                    attention_seq_len_2 * attention_seq_len_1,
                                                    request_batch_size * local_head_num_,  // global batch size
                                                    CUDA_R_32F);

                sync_check_cuda_error();
                POP_RANGE;

                PUSH_RANGE("softmax");
                MaskedSoftmaxParam<T, float> param;
                param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.qk                 = qk_buf_float_;   // (batch_size, head_num, q_length, k_length)
                param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
                param.batch_size         = request_batch_size;
                param.q_length           = attention_seq_len_1;
                param.k_length           = attention_seq_len_2;
                param.num_heads          = local_head_num_;
                param.qk_scale           = qk_scale;
                param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
                printf("[%s:%d-pid:%d-%s]: <invokeMaskedSoftmax>  \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__);
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
                POP_RANGE;
            }
            else {
                PUSH_RANGE("Q*K batch gemm");
                printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->stridedBatchedGemm> [Q*K batch gemm] m%dn%dk%d batch_size=%d, q_length=%d, k_length=%d, num_heads=%d\n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    attention_seq_len_1, attention_seq_len_2, size_per_head_,
                    request_batch_size,//1
                    attention_seq_len_1,//8
                    attention_seq_len_2,//8
                    local_head_num_);//16
                cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    attention_seq_len_2,//m 8
                                                    attention_seq_len_1,//n 8
                                                    size_per_head_,//k 64
                                                    k_buf_2_,//A
                                                    size_per_head_,//lda
                                                    attention_seq_len_2 * size_per_head_,//stride A
                                                    q_buf_2_,//B
                                                    size_per_head_,//ldb
                                                    attention_seq_len_1 * size_per_head_,//stride B
                                                    qk_buf_,//C
                                                    attention_seq_len_2,//ldc
                                                    attention_seq_len_2 * attention_seq_len_1,//stride C
                                                    request_batch_size * local_head_num_);//batch_size 1*16

                POP_RANGE;
                PUSH_RANGE("softmax");
                MaskedSoftmaxParam<T, T> param;
                param.attention_score    = qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.qk                 = qk_buf_;         // (batch_size, head_num, q_length, k_length)
                param.attention_mask     = attention_mask;  // (batch_size, q_length, k_length)
                param.batch_size         = request_batch_size;  // 1
                param.q_length           = attention_seq_len_1; // 8
                param.k_length           = attention_seq_len_2; // 8
                param.num_heads          = local_head_num_;     // 16
                param.qk_scale           = qk_scale;
                param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes);  // (head_num,), optional
                printf("[%s:%d-pid:%d-%s]: <invokeMaskedSoftmax>  \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__);
                {
                    int out_size = param.batch_size*param.q_length*param.k_length; //batch_size * beam_width * max_input_length * max_input_length;
                    std::vector<T> h_buf(out_size);
                    cudaD2Hcpy(h_buf.data(), attention_mask, out_size);
                    printf("-------invokeMaskedSoftmax [softmax] attention_mask[%d]%d/%d/%d:\n", 
                        out_size,sizeof(T),sizeof(bool),sizeof(int16_t));
                    for(int k=0; k<out_size; k++) {
                        printf("%d ", (int32_t)h_buf[k]);
                    }
                    printf("\n");
                }
                invokeMaskedSoftmax(param, stream_);
                sync_check_cuda_error();
                POP_RANGE;
            }

            PUSH_RANGE("QK*V batch gemm");
            printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->stridedBatchedGemm> [QK*V batch gemm] m%dn%dk%d A[%d,%d] B[%d,%d] C[%d,%d] batch:%d\n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    size_per_head_,attention_seq_len_1,attention_seq_len_2,
                    size_per_head_,attention_seq_len_2 * size_per_head_,
                    attention_seq_len_2,attention_seq_len_1 * attention_seq_len_2,
                    size_per_head_,attention_seq_len_1 * size_per_head_,
                    request_batch_size * local_head_num_);
            cublas_wrapper_->stridedBatchedGemm(CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                size_per_head_,//m  64
                                                attention_seq_len_1,//n 8 
                                                attention_seq_len_2,//k 8
                                                v_buf_2_,//A
                                                size_per_head_,//lda  64
                                                attention_seq_len_2 * size_per_head_,//strideA 8*64=512
                                                qk_buf_,//B
                                                attention_seq_len_2,//ldb 8
                                                attention_seq_len_1 * attention_seq_len_2,//strideB 8*8=64 
                                                qkv_buf_2_,//C
                                                size_per_head_,//ldc 64
                                                attention_seq_len_1 * size_per_head_,//strideC 8*64=512
                                                request_batch_size * local_head_num_);//batch_size=

            // transpose (batch_size, num_heads, L, Dh) to (batch_size, L, num_heads * Dh)
            if (padding_offset == nullptr) {
                printf("[%s:%d-pid:%d-%s]: <invokeTransposeQKV>  request_batch_size:%d attention_seq_len_1:%d local_head_num_:%d size_per_head_:%d\n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    request_batch_size, attention_seq_len_1, local_head_num_, size_per_head_);
                invokeTransposeQKV(qkv_buf_3_,
                                   qkv_buf_2_,
                                   request_batch_size,
                                   attention_seq_len_1,
                                   local_head_num_,
                                   size_per_head_,
                                   attention_weights->attention_output_weight.scale,
                                   int8_mode_,
                                   stream_);
                sync_check_cuda_error();
            }
            else {
                printf("[%s:%d-pid:%d-%s]: <invokeTransposeAttentionOutRemovePadding> scale=%p m%d %d %d %d %d\n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    attention_weights->attention_output_weight.scale,
                    m,//8
                    request_batch_size,//1
                    attention_seq_len_1,//8
                    local_head_num_,//16
                    size_per_head_);//64
                invokeTransposeAttentionOutRemovePadding(qkv_buf_2_,//in [QK*V] [1,16,64,8]
                                                         qkv_buf_3_,//out [1,1024,8] 
                                                         m,//8
                                                         request_batch_size,//1
                                                         attention_seq_len_1,//8
                                                         local_head_num_,//16
                                                         size_per_head_,//64
                                                         padding_offset,
                                                         attention_weights->attention_output_weight.scale,
                                                         int8_mode_,
                                                         stream_);
            }
            POP_RANGE;
        }
        sync_check_cuda_error();

        PUSH_RANGE("proj gemm");
#ifdef SPARSITY_ENABLED
        bool use_sparse = sparse_ && cublas_wrapper_->isUseSparse(1, hidden_units_, m_padded, local_hidden_units_);
#endif

        if (use_sparse) {
#ifdef SPARSITY_ENABLED
            cublas_wrapper_->SpGemm(CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    hidden_units_,
                                    m_padded,
                                    local_hidden_units_,
                                    attention_weights->attention_output_weight.sp_kernel,
                                    qkv_buf_3_,
                                    attention_out);
#endif
        }
        else {
            if (int8_mode_ == 1) {
                FT_CHECK(weight_only_int8_fc_runner_.get() != NULL
                         && attention_weights->attention_output_weight.int8_kernel != NULL
                         && attention_weights->attention_output_weight.weight_only_quant_scale != NULL);

                printf("[%s:%d-pid:%d-%s]: <weight_only_int8_fc_runner_->gemm>  \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__);
                weight_only_int8_fc_runner_->gemm(
                    qkv_buf_3_,
                    reinterpret_cast<const uint8_t*>(attention_weights->attention_output_weight.int8_kernel),
                    attention_weights->attention_output_weight.weight_only_quant_scale,
                    attention_out,
                    m,
                    hidden_units_,
                    local_hidden_units_,
                    mixed_gemm_workspace_,
                    mixed_gemm_ws_bytes_,
                    stream_);
            }
            else if (int8_mode_ == 2) {
                printf("[%s:%d-pid:%d-%s]: <int8_fc_runner_->gemm>  \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__);
                int8_fc_runner_->gemm(reinterpret_cast<int8_t*>(qkv_buf_3_),
                                      attention_weights->attention_output_weight.int8_kernel,
                                      QuantMode::PerTensorQuant,
                                      attention_weights->attention_output_weight.scale_inter,
                                      attention_weights->attention_output_weight.scale_out,
                                      output_tensors->at("hidden_features").getPtr<T>(),
                                      m,
                                      hidden_units_,
                                      local_hidden_units_,
                                      nullptr,
                                      0,
                                      stream_);
            }
            else {
                printf("[%s:%d-pid:%d-%s]: <cublas_wrapper_->Gemm> m%dn%dk%d \n", 
                    __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                    hidden_units_,m,local_hidden_units_);
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      hidden_units_,//1024
                                      m,//8
                                      local_hidden_units_,//1024
                                      attention_weights->attention_output_weight.kernel,//A
                                      hidden_units_,//1024
                                      qkv_buf_3_,//B
                                      local_hidden_units_,
                                      attention_out,
                                      hidden_units_);
            }
        }
        POP_RANGE;
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(0),
    neox_rotary_style_(false),
    is_qk_buf_float_(is_qk_buf_float || int8_mode == 2),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(size_t           max_batch_size,
                                                      size_t           max_seq_len,
                                                      size_t           head_num,
                                                      size_t           size_per_head,
                                                      size_t           local_head_num,
                                                      size_t           rotary_embedding_dim,
                                                      bool             neox_rotary_style,
                                                      cudaStream_t     stream,
                                                      cublasMMWrapper* cublas_wrapper,
                                                      IAllocator*      allocator,
                                                      bool             is_free_buffer_after_forward,
                                                      bool             is_qk_buf_float,
                                                      bool             sparse,
                                                      int              int8_mode):
    BaseAttentionLayer<T>(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, sparse),
    max_batch_size_(max_batch_size),
    max_seq_len_(max_seq_len),
    head_num_(head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    local_head_num_(local_head_num),
    local_hidden_units_(local_head_num_ * size_per_head),
    rotary_embedding_dim_(rotary_embedding_dim),
    neox_rotary_style_(neox_rotary_style),
    is_qk_buf_float_(is_qk_buf_float),
    weight_only_int8_fc_runner_(int8_mode == 1 ? std::make_shared<CutlassFpAIntBGemmRunner<T, uint8_t>>() : nullptr),
    int8_fc_runner_(int8_mode == 2 ? std::make_shared<CutlassInt8GemmRunner<T>>() : nullptr),
    int8_mode_(int8_mode)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatcher_fp16.reset(new FusedMHARunnerFP16v2(local_head_num_, size_per_head_, sm_, 1.0f));
}

template<typename T>
GptContextAttentionLayer<T>::GptContextAttentionLayer(GptContextAttentionLayer<T> const& attention_layer):
    BaseAttentionLayer<T>(attention_layer.stream_,
                          attention_layer.cublas_wrapper_,
                          attention_layer.allocator_,
                          attention_layer.is_free_buffer_after_forward_,
                          attention_layer.sparse_),
    max_batch_size_(attention_layer.max_batch_size_),
    max_seq_len_(attention_layer.max_seq_len_),
    head_num_(attention_layer.head_num_),
    size_per_head_(attention_layer.size_per_head_),
    hidden_units_(attention_layer.hidden_units_),
    local_head_num_(attention_layer.local_head_num_),
    local_hidden_units_(attention_layer.local_hidden_units_),
    rotary_embedding_dim_(attention_layer.rotary_embedding_dim_),
    neox_rotary_style_(attention_layer.neox_rotary_style_),
    is_qk_buf_float_(attention_layer.is_qk_buf_float_),
    weight_only_int8_fc_runner_(attention_layer.weight_only_int8_fc_runner_),
    int8_fc_runner_(attention_layer.int8_fc_runner_),
    int8_mode_(attention_layer.int8_mode_)
{
}

template<typename T>
GptContextAttentionLayer<T>::~GptContextAttentionLayer()
{
    cublas_wrapper_ = nullptr;
    freeBuffer();
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void GptContextAttentionLayer<T>::allocateBuffer(size_t batch_size, size_t seq_len, bool allocate_qk_buf)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    // const auto type_size = int8_mode_ == 2 ? sizeof(int8_t) : sizeof(T);
    // NOTE (perkzz): use sizeof(T) here for cutlass int8 kernels.
    const auto type_size = sizeof(T);
    printf("[%s:%d-pid:%d-%s]: <qkv_buf_:%d/%d>  batch_size=%d seq_len=%d local_hidden_units_=%d\n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__,
        type_size * 3 * batch_size * seq_len * local_hidden_units_,//49152  24576=3*1(batch_size)*8(seq_len)*1024(local_hidden_units_)
        3 * batch_size * seq_len * local_hidden_units_,//24576
        batch_size, //1
        seq_len, //8
        local_hidden_units_);//1024
    qkv_buf_ = (T*)allocator_->reMalloc(qkv_buf_, type_size * 3 * batch_size * seq_len * local_hidden_units_, true);
    printf("[%s:%d-pid:%d-%s]: <q_buf_2_:%d/%d>  batch_size=%d seq_len=%d local_hidden_units_=%d\n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__,
        sizeof(T) * batch_size * seq_len * 3 * local_hidden_units_,//49152
        batch_size * seq_len * 3 * local_hidden_units_,//24576
        batch_size, seq_len, local_hidden_units_);// 1 8 1024
    q_buf_2_ = (T*)allocator_->reMalloc(q_buf_2_, sizeof(T) * batch_size * seq_len * 3 * local_hidden_units_, true);
    k_buf_2_ = q_buf_2_ + batch_size * seq_len * local_hidden_units_;
    v_buf_2_ = k_buf_2_ + batch_size * seq_len * local_hidden_units_;
    printf("[%s:%d-pid:%d-%s]: <q_buf_2_:%d k_buf_2_:%d v_buf_2_:%d>\n", 
        __FUNCTION__, __LINE__, getpid(),__FILENAME__,
        batch_size * seq_len * local_hidden_units_,//8192  batch_size:1 seq_len:8 local_hidden_units_:1024
        batch_size * seq_len * local_hidden_units_,//8192
        batch_size * seq_len * local_hidden_units_);//8192

    // save memory usage when using fmha
    if (allocate_qk_buf) {
        printf("[%s:%d-pid:%d-%s]: <qk_buf_:%d/%d>  batch_size=%d local_head_num_=%d seq_len=%d\n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__,
            sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len,//2048
            batch_size * local_head_num_ * seq_len * seq_len,//1024
            batch_size, local_head_num_, seq_len);// 1  16 8
        qk_buf_ = (T*)allocator_->reMalloc(qk_buf_, sizeof(T) * batch_size * local_head_num_ * seq_len * seq_len, true);
    }
    else {
        allocator_->free((void**)(&qk_buf_));
    }
    printf("[%s:%d-pid:%d-%s]: <qkv_buf_2_:%d/%d>  batch_size=%d seq_len=%d local_hidden_units_=%d\n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__,
            sizeof(T) * batch_size * seq_len * local_hidden_units_,//16384
            batch_size * seq_len * local_hidden_units_,//8192
            batch_size, seq_len, local_hidden_units_);// 1 seq_len=8 local_hidden_units_=1024
    qkv_buf_2_ = (T*)allocator_->reMalloc(qkv_buf_2_, sizeof(T) * batch_size * seq_len * local_hidden_units_, true);
    printf("[%s:%d-pid:%d-%s]: <qkv_buf_3_:%d/%d>  batch_size=%d seq_len=%d local_hidden_units_=%d\n", 
            __FUNCTION__, __LINE__, getpid(),__FILENAME__,
            type_size * batch_size * seq_len * local_hidden_units_,//16384
            batch_size * seq_len * local_hidden_units_,//8192
            batch_size, seq_len, local_hidden_units_);// 1 8 1024
    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, type_size * batch_size * seq_len * local_hidden_units_, true);

    if (is_qk_buf_float_ == true) {
        if (allocate_qk_buf) {
            printf("[%s:%d-pid:%d-%s]: <qk_buf_float_:%d/%d>  batch_size=%d local_head_num_=%d seq_len=%d\n", 
                __FUNCTION__, __LINE__, getpid(),__FILENAME__,
                sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len,
                batch_size * local_head_num_ * seq_len * seq_len,
                batch_size, local_head_num_, seq_len);
            qk_buf_float_ = (float*)allocator_->reMalloc(
                qk_buf_float_, sizeof(float) * batch_size * local_head_num_ * seq_len * seq_len, true);
        }
        else {
            allocator_->free((void**)(&qk_buf_float_));
        }
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(hidden_units_, 3 * local_hidden_units_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }

    if (int8_mode_ == 1) {
        // We use max_size for n and k since we reuse buffers for both FCs and want to allocate the max
        // possible memory that would be required by any of the individual gemms.
        const int max_size    = std::max(hidden_units_, 3 * local_hidden_units_);
        mixed_gemm_ws_bytes_  = weight_only_int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        mixed_gemm_workspace_ = (char*)allocator_->reMalloc(mixed_gemm_workspace_, mixed_gemm_ws_bytes_, false);
    }
    else if (int8_mode_ == 2) {
        const int max_size   = std::max(hidden_units_, 3 * local_hidden_units_);
        int8_gemm_ws_bytes_  = int8_fc_runner_->getWorkspaceSize(batch_size * seq_len, max_size, max_size);
        int8_gemm_workspace_ = (char*)allocator_->reMalloc(int8_gemm_workspace_, int8_gemm_ws_bytes_, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void GptContextAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        FT_LOG_DEBUG(__PRETTY_FUNCTION__);
        allocator_->free((void**)(&qkv_buf_));
        allocator_->free((void**)(&q_buf_2_));
        allocator_->free((void**)(&qk_buf_));
        allocator_->free((void**)(&qkv_buf_2_));
        allocator_->free((void**)(&qkv_buf_3_));

        if (is_qk_buf_float_ == true) {
            allocator_->free((void**)(&qk_buf_float_));
        }

        allocator_->free((void**)(&mixed_gemm_workspace_));
        mixed_gemm_ws_bytes_ = 0;

        allocator_->free((void**)(&int8_gemm_workspace_));
        int8_gemm_ws_bytes_ = 0;

        is_allocate_buffer_ = false;
    }
}

template class GptContextAttentionLayer<float>;
template class GptContextAttentionLayer<half>;
#ifdef ENABLE_BF16
template class GptContextAttentionLayer<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
