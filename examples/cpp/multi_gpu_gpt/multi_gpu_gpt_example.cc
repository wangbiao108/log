/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "3rdparty/INIReader.h"
#include "examples/cpp/multi_gpu_gpt/gpt_example_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"

#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <tuple>
#include <vector>

using namespace fastertransformer;

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv);

int main(int argc, char* argv[])
{
    mpi::initialize(&argc, &argv);
    srand(0);

    std::string ini_name;
    if (argc >= 2) {
        ini_name = std::string(argv[1]);
    }
    else {
        ini_name = "../examples/cpp/multi_gpu_gpt/gpt_config.ini";
    }

    std::string in_csv;
    if (argc == 3) {
        in_csv = std::string(argv[2]);
    }
    else {
        in_csv = "../examples/cpp/multi_gpu_gpt/start_ids.csv";
    }

    INIReader reader = INIReader(ini_name);
    if (reader.ParseError() < 0) {
        std::cout << "[ERROR] Can't load '" << ini_name << "'\n";
        return -1;
    }
    const std::string data_type = reader.Get("ft_instance_hyperparameter", "data_type");

    if (data_type == "fp32") {
        multi_gpu_gpt_example<float>(reader, in_csv);
    }
    else if (data_type == "fp16") {
        multi_gpu_gpt_example<half>(reader, in_csv);
    }
#ifdef ENABLE_BF16
    else if (data_type == "bf16") {
        multi_gpu_gpt_example<__nv_bfloat16>(reader, in_csv);
    }
#endif
    else {
        printf("[ERROR] data_type should be fp32, fp16 or bf16 ! \n");
        return -1;
    }
    mpi::finalize();
    return 0;
}

template<typename T>
void multi_gpu_gpt_example(const INIReader reader, std::string in_csv)
{
    const DataType data_type = getTensorType<T>();

    auto                                       model_config = read_model_config(reader);
    std::map<std::string, std::pair<int, int>> p_prompt_tuning_table_pair(init_prompt_tuning_map(reader, model_config));
    auto                                       request_config = read_request_config(reader);

    printf("**************************************************************\n");
    printf("model config: \n");
    printf("model_name                : %s\n", model_config.model_name.c_str());
    printf("model_dir                 : %s\n", model_config.model_dir.c_str());
    printf("sparse                    : %d\n", model_config.sparse);
    printf("int8_mode                 : %d\n", model_config.int8_mode);
    printf("attention_type            : %d\n", model_config.attention_type);
    printf("tensor_para_size          : %d\n", model_config.tensor_para_size);
    printf("pipeline_para_size        : %d\n", model_config.pipeline_para_size);
    printf("head_num                  : %d\n", (int32_t)model_config.head_num);
    printf("size_per_head             : %d\n", (int32_t)model_config.size_per_head);
    printf("vocab_size                : %d\n", (int32_t)model_config.vocab_size);
    printf("decoder_layers            : %d\n", (int32_t)model_config.decoder_layers);
    printf("hidden_units              : %d\n", (int32_t)model_config.hidden_units);
    printf("inter_size                : %d\n", (int32_t)model_config.inter_size);
    printf("max_seq_len               : %d\n", (int32_t)model_config.max_seq_len);
    printf("start_id                  : %d\n", (int32_t)model_config.start_id);
    printf("end_id                    : %d\n", (int32_t)model_config.end_id);
    printf("prompt_learning_start_id  : %d\n", (int32_t)model_config.prompt_learning_start_id);
    printf("prompt_learning_num_tasks : %d\n", (int32_t)model_config.prompt_learning_num_tasks);
    printf("prompt_learning_type      : %d\n", (int32_t)model_config.prompt_learning_type);
    printf("gpt_variants: \n");
    printf("layernorm_eps             : %f\n", model_config.gpt_variants.layernorm_eps);
    printf("layernorm_type            : %d\n", (int32_t)model_config.gpt_variants.layernorm_type);
    printf("activation_type           : %d\n", (int32_t)model_config.gpt_variants.activation_type);
    printf("has_positional_encoding   : %d\n", (int32_t)model_config.gpt_variants.has_positional_encoding);
    printf("has_pre_decoder_layernorm : %d\n", (int32_t)model_config.gpt_variants.has_pre_decoder_layernorm);
    printf("has_post_decoder_layernorm: %d\n", (int32_t)model_config.gpt_variants.has_post_decoder_layernorm);
    printf("has_adapters              : %d\n", (int32_t)model_config.gpt_variants.has_adapters);
    printf("adapter_inter_size        : %d\n", (int32_t)model_config.gpt_variants.adapter_inter_size);
    printf("use_attention_linear_bias : %d\n", (int32_t)model_config.gpt_variants.use_attention_linear_bias);
    printf("\n");

    printf("init prompt tuning map: \n");
    printf("\n");

    printf("request config: \n");
    printf("request_batch_size              : %d\n", (int32_t)request_config.request_batch_size);
    printf("request_output_len              : %d\n", (int32_t)request_config.request_output_len);
    printf("is_return_log_probs             : %d\n", (int32_t)request_config.is_return_log_probs);
    printf("is_return_context_cum_log_probs : %d\n", (int32_t)request_config.is_return_context_cum_log_probs);
    printf("is_return_context_embeddings    : %d\n", (int32_t)request_config.is_return_context_embeddings);
    printf("remove_padding                  : %d\n", (int32_t)request_config.remove_padding);
    printf("memory_len                      : %d\n", (int32_t)request_config.memory_len);
    printf("beam_width                      : %d\n", (int32_t)request_config.beam_width);
    printf("top_k                           : %d\n", (int32_t)request_config.top_k);
    printf("top_p                           : %f\n", request_config.top_p);
    printf("temperature                     : %f\n", request_config.temperature);
    printf("repetition_penalty              : %f\n", request_config.repetition_penalty);
    printf("presence_penalty                : %f\n", request_config.presence_penalty);
    printf("min_length                      : %d\n", (int32_t)request_config.min_length);
    printf("len_penalty                     : %f\n", request_config.len_penalty);
    printf("beam_search_diversity_rate      : %f\n", request_config.beam_search_diversity_rate);
    printf("shared_contexts_ratio           : %f\n", request_config.shared_contexts_ratio);
    printf("start_id                        : %d\n", (int32_t)request_config.start_id);
    printf("end_id                          : %d\n", (int32_t)request_config.end_id);
    printf("\n");

    // printf("max_batch_size            : %d\n", model_config.max_batch_size);
    // printf("max_seq_len               : %d\n", model_config.max_seq_len);

    printf("**************************************************************\n");
    printf("\n");

    NcclParam tensor_para, pipeline_para;
    int       rank;//当前进程的序号
    int       world_size;//总共运行进程个数
    std::tie(rank, world_size) = init_multiprocessing(model_config);

    printf("multi_gpu_gpt_example: rank=%d, world_size=%d\n", rank, world_size);

    cudaStream_t                   stream;
    cudaDeviceProp                 prop;
    Allocator<AllocatorType::CUDA> allocator(init_cuda_ctx(stream, prop, rank));

    cublasHandle_t   cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    std::mutex       cublas_wrapper_mutex;
    cublasAlgoMap    cublas_algo_map;

#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparselt_handle;
#endif

    init_nccl(model_config, tensor_para, pipeline_para);
    cublasMMWrapper cublas_wrapper = init_cublas_ctx(data_type,
                                                     stream,
                                                     allocator,
                                                     cublas_handle,
                                                     cublas_wrapper_mutex,
                                                     cublaslt_handle,
                                                     cublas_algo_map
#ifdef SPARSITY_ENABLED 
                                                     ,
                                                     cusparselt_handle
#endif
    );

    printf("ParallelGptWeight: tensor_para.rank_=%d, pipeline_para.rank_=%d; tensor_para.world_size_=%d,pipeline_para.world_size_=%d\n", 
        tensor_para.rank_, pipeline_para.rank_,
        tensor_para.world_size_,pipeline_para.world_size_);
    printf("-----------权重配置生成\n");
    ParallelGptWeight<T> gpt_weights(model_config.hidden_units,
                                     model_config.inter_size,
                                     model_config.vocab_size,
                                     model_config.decoder_layers,
                                     model_config.max_seq_len,
                                     tensor_para.world_size_,
                                     tensor_para.rank_,
                                     pipeline_para.world_size_,
                                     pipeline_para.rank_,
                                     model_config.int8_mode,
                                     model_config.prompt_learning_type,
                                     p_prompt_tuning_table_pair,
                                     model_config.gpt_variants);
    printf("------------权重加载\n");
    gpt_weights.loadModel(model_config.model_dir);
    printf("------------权重加载完成\n");

    printf("gpt weight load model finish\n");
#ifdef SPARSITY_ENABLED
    if (model_config.sparse) {
        printf("[INFO] Compress weights for sparse inference\n");
        gpt_weights.compress_weights(cublas_wrapper);
    }
#endif

    unsigned long long random_seed;
    if (rank == 0) {
        random_seed = (unsigned long long)(0);
    }
    if (world_size > 1) {
        mpi::bcast(&random_seed, 1, mpi::MPI_TYPE_UNSIGNED_LONG_LONG, 0, mpi::COMM_WORLD);
    }

    AttentionType attention_type = getAttentionType<T>(model_config.size_per_head,
                                                       getSMVersion(),
                                                       request_config.remove_padding,  // remove_padding
                                                       0,                            // gpt supports any-seq-length fmha
                                                       model_config.int8_mode != 2,  // is_fuse
                                                       false,                        // with_relative_position_bias
                                                       true);                        // causal_mask

    printf("ParallelGpt\n");
    ParallelGpt<T> gpt = ParallelGpt<T>(0,  // max_batch_size, FT will adjust the buffer automatically.
                                        0,  // max_seq_len, FT will adjust the buffer automatically.
                                        0,  // max_input_len, FT will adjust the buffer automatically.
                                        0,  // beam_width, FT will adjust the buffer automatically.
                                        model_config.head_num,
                                        model_config.size_per_head,
                                        model_config.inter_size,
                                        model_config.decoder_layers,
                                        0,   // expert_num
                                        0,   // moe_k
                                        {},  // moe_layer_index
                                        model_config.vocab_size,
                                        model_config.start_id,
                                        model_config.end_id,
                                        // p/prompt tuning virtual token start id
                                        model_config.prompt_learning_start_id,
                                        model_config.prompt_learning_type,
                                        model_config.gpt_variants,
                                        0.0f,  // beam_search_diversity_rate,
                                        0,     // top_k,
                                        0.0,   // top_p,
                                        0,     // random_seed,
                                        1.0f,  // temperature,
                                        0.0f,  // len_penalty,
                                        1.0f,  // repetition_penalty,
                                        tensor_para,
                                        pipeline_para,
                                        stream,
                                        &cublas_wrapper,
                                        &allocator,
                                        false,
                                        &prop,
                                        attention_type,
                                        model_config.sparse,
                                        model_config.int8_mode,
                                        nullptr,
                                        0,
                                        request_config.shared_contexts_ratio);

    std::unordered_map<std::string, Tensor> input_tensors;
    int*                                    d_input_ids     = nullptr;
    int*                                    d_input_lengths = nullptr;
    std::vector<int>                        v_start_lengths;
    std::vector<int>                        output_seq_len;
    std::vector<int>                        p_prompt_tuning_task_ids;

    std::unordered_map<std::string, Tensor> output_tensors;
    int*                                    d_output_ids                = nullptr;
    int*                                    d_sequence_lengths          = nullptr;
    float*                                  d_output_log_probs          = nullptr;
    float*                                  d_cum_log_probs             = nullptr;
    float*                                  d_output_context_embeddings = nullptr;

    printf("populate_request\n");
    populate_request(input_tensors,
                     output_tensors,
                     d_input_ids,
                     d_input_lengths,
                     v_start_lengths,
                     output_seq_len,
                     p_prompt_tuning_task_ids,
                     d_output_ids,
                     d_sequence_lengths,
                     d_output_log_probs,
                     d_cum_log_probs,
                     d_output_context_embeddings,
                     random_seed,
                     in_csv,
                     model_config,
                     request_config);

    printf("print_mem_usage\n");
    print_mem_usage();

    int ite = 1;
    printf("cudaDeviceSynchronize\n");
    cudaDeviceSynchronize();
    mpi::barrier();

    printf("cudaProfilerStart\n");
    cudaProfilerStart();
    // warm up
    ite = 1;

    ft_nvtx::setScope("warmup_time");
    PUSH_RANGE("warmup time")
    for (int i = 0; i < ite; ++i) {
        gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    }
    printf("cudaDeviceSynchronize2,ite=%d\n",ite);
    cudaDeviceSynchronize();
    mpi::barrier();

    POP_RANGE;
    ft_nvtx::resetScope();

    if (rank == 0) {
        write_output_tensors(output_tensors);
    }
    // test time
    printf("cudaDeviceSynchronize3\n");
    struct timeval start, end;
    mpi::barrier();
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);

    // ite = 10;

    // for (int i = 0; i < ite; ++i) {
    //     PUSH_RANGE("batch");
    //     gpt.forward(&output_tensors, &input_tensors, &gpt_weights);
    //     POP_RANGE;
    // }

    printf("cudaDeviceSynchronize4\n");
    cudaDeviceSynchronize();
    mpi::barrier();

    gettimeofday(&end, NULL);

    printf("cudaProfilerStop\n");
    cudaProfilerStop();

    const auto total_output_len = output_tensors.at("output_ids").shape[2];
    printf("[INFO] request_batch_size %ld beam_width %ld head_num %ld size_per_head %ld total_output_len %ld"
           " decoder_layers %ld vocab_size %ld FT-CPP-decoding-beamsearch-time %.2f ms\n",
           request_config.request_batch_size,
           request_config.beam_width,
           model_config.head_num,
           model_config.size_per_head,
           total_output_len,
           model_config.decoder_layers,
           model_config.vocab_size,
           ((end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001) / ite);

    printf("safe_free\n");
    safe_free(d_input_ids);
    safe_free(d_input_lengths);
    safe_free(d_output_ids);
    safe_free(d_sequence_lengths);
    safe_free(d_output_log_probs);
    safe_free(d_cum_log_probs);
    safe_free(d_output_context_embeddings);

    printf("ftNcclParamDestroy\n");
    ftNcclParamDestroy(tensor_para);
    ftNcclParamDestroy(pipeline_para);

#ifdef SPARSITY_ENABLED
    cusparseLtDestroy(&cusparselt_handle);
#endif
}
