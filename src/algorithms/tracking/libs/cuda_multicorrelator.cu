/*!
 * \file cuda_multicorrelator.cu
 * \brief Fixed CUDA GPU vector multiTAP correlator — split kernel version
 *
 * Original author: Javier Arribas, 2015. jarribas(at)cttc.es
 * Bug fix: Split fused kernel to eliminate inter-block race condition on
 * d_sig_wiped[], and fix off-by-one in negative chip index wrapping.
 *
 * BUG 1 (race condition): The original Doppler_wippe_scalarProdGPUCPXxN_shifts_chips
 * kernel fuses Doppler wipe-off (writes d_sig_wiped[]) and correlation
 * (reads d_sig_wiped[]) into a single kernel with only __syncthreads()
 * between them. __syncthreads() only synchronizes within a block, so
 * with blocksPerGrid=128, block 0 can read d_sig_wiped[] while block 5
 * is still writing it.
 *
 * FIX: Split into two kernels launched sequentially on the same stream.
 * CUDA stream ordering guarantees kernel 2 sees all writes from kernel 1.
 *
 * BUG 2 (off-by-one): Original line 96 uses (code_length_chips - 1) for
 * negative wrap-around, but fmodf can return values in (-code_length_chips, 0),
 * so the correct addition is code_length_chips (not code_length_chips - 1).
 *
 * -----------------------------------------------------------------------------
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 *
 * GNSS-SDR is a software defined Global Navigation
 *          Satellite Systems receiver
 *
 * This file is part of GNSS-SDR.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */

#include "cuda_multicorrelator.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#define ACCUM_N 128


/*
 * Kernel 1: Doppler wipe-off
 *
 * Generates a local carrier replica via sincosf NCO and multiplies it
 * with the input signal, storing the result in d_sig_wiped[].
 * This is the write-only kernel — no reads from d_sig_wiped[].
 */
__global__ void doppler_wipeoff_kernel(
    GPU_Complex *d_sig_in,
    GPU_Complex *d_sig_wiped,
    int elementN,
    float rem_carrier_phase_in_rad,
    float phase_step_rad)
{
    float sin_val;
    float cos_val;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < elementN;
         i += blockDim.x * gridDim.x)
        {
            __sincosf(rem_carrier_phase_in_rad + i * phase_step_rad,
                      &sin_val, &cos_val);
            d_sig_wiped[i] = d_sig_in[i] * GPU_Complex(cos_val, -sin_val);
        }
}


/*
 * Kernel 2: Multi-tap correlation with integrated code resampling
 *
 * Reads the Doppler-wiped signal from d_sig_wiped[] (written by kernel 1),
 * resamples the local code for each correlator tap shift, and accumulates
 * the complex dot products via shared-memory tree reduction.
 *
 * SAFETY: Because this kernel launches AFTER doppler_wipeoff_kernel on the
 * same stream, all writes to d_sig_wiped[] are guaranteed complete before
 * any thread in this kernel reads from it.
 */
__global__ void correlation_kernel(
    GPU_Complex *d_corr_out,
    GPU_Complex *d_sig_wiped,
    GPU_Complex *d_local_code_in,
    float *d_shifts_chips,
    int code_length_chips,
    float code_phase_step_chips,
    float rem_code_phase_chips,
    int vectorN,
    int elementN)
{
    // Per-block accumulator array in shared memory
    __shared__ GPU_Complex accumResult[ACCUM_N];

    // Cycle through correlator taps (vectorN = n_correlators).
    for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x)
        {
            for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
                {
                    GPU_Complex sum = GPU_Complex(0, 0);
                    float local_code_chip_index = 0.0;

                    for (int pos = iAccum; pos < elementN; pos += ACCUM_N)
                        {
                            local_code_chip_index = fmodf(
                                code_phase_step_chips * __int2float_rd(pos)
                                + d_shifts_chips[vec]
                                - rem_code_phase_chips,
                                code_length_chips);

                            // FIX: correct wrap-around for negative chip indices.
                            // Original used (code_length_chips - 1) which is wrong:
                            // fmodf can return values in (-code_length_chips, 0),
                            // so we must add the full code_length_chips to map
                            // back into [0, code_length_chips).
                            if (local_code_chip_index < 0.0f)
                                local_code_chip_index += code_length_chips;

                            sum.multiply_acc(d_sig_wiped[pos],
                                d_local_code_in[__float2int_rd(local_code_chip_index)]);
                        }
                    accumResult[iAccum] = sum;
                }

            // Tree reduction
            for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1)
                {
                    __syncthreads();

                    for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
                        {
                            accumResult[iAccum] += accumResult[stride + iAccum];
                        }
                }

            if (threadIdx.x == 0)
                {
                    d_corr_out[vec] = accumResult[0];
                }
        }
}


/*
 * Host-side launch function (replaces the original single-kernel launch).
 *
 * The two kernels are launched sequentially on the same CUDA stream.
 * The CUDA runtime guarantees that kernel 2 will not begin execution
 * until kernel 1 has completed on ALL blocks.
 */
bool cuda_multicorrelator::Carrier_wipeoff_multicorrelator_resampler_cuda(
    float rem_carrier_phase_in_rad,
    float phase_step_rad,
    float code_phase_step_chips,
    float rem_code_phase_chips,
    int signal_length_samples,
    int n_correlators)
{
    cudaSetDevice(selected_gps_device);

    // Kernel 1: Doppler wipe-off (writes d_sig_doppler_wiped)
    doppler_wipeoff_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
        d_sig_in,
        d_sig_doppler_wiped,
        signal_length_samples,
        rem_carrier_phase_in_rad,
        phase_step_rad);

    // Kernel 2: Correlation accumulation (reads d_sig_doppler_wiped)
    // Implicit stream ordering guarantees all wipe-off writes are visible.
    correlation_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
        d_corr_out,
        d_sig_doppler_wiped,
        d_local_codes_in,
        d_shifts_chips,
        d_code_length_chips,
        code_phase_step_chips,
        rem_code_phase_chips,
        n_correlators,
        signal_length_samples);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream1));

    return true;
}


#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
}


bool cuda_multicorrelator::init_cuda_integrated_resampler(
    int signal_length_samples,
    int code_length_chips,
    int n_correlators)
{
    cudaDeviceProp prop;
    int num_devices, device;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1)
        {
            int max_multiprocessors = 0, max_device = 0;
            for (device = 0; device < num_devices; device++)
                {
                    cudaDeviceProp properties;
                    cudaGetDeviceProperties(&properties, device);
                    if (max_multiprocessors < properties.multiProcessorCount)
                        {
                            max_multiprocessors = properties.multiProcessorCount;
                            max_device = device;
                        }
                    printf("Found GPU device # %i\n", device);
                }

            selected_gps_device = rand() % num_devices;
            cudaSetDevice(selected_gps_device);

            cudaGetDeviceProperties(&prop, max_device);
            if (prop.canMapHostMemory != 1)
                {
                    printf("Device can not map memory.\n");
                }
            printf("L2 Cache size= %u \n", prop.l2CacheSize);
            printf("maxThreadsPerBlock= %u \n", prop.maxThreadsPerBlock);
            printf("maxGridSize= %i \n", prop.maxGridSize[0]);
            printf("sharedMemPerBlock= %lu \n", prop.sharedMemPerBlock);
            printf("deviceOverlap= %i \n", prop.deviceOverlap);
            printf("multiProcessorCount= %i \n", prop.multiProcessorCount);
        }
    else
        {
            cudaGetDevice(&selected_gps_device);
            cudaGetDeviceProperties(&prop, selected_gps_device);
            if (prop.canMapHostMemory != 1)
                {
                    printf("Device can not map memory.\n");
                }

            printf("L2 Cache size= %u \n", prop.l2CacheSize);
            printf("maxThreadsPerBlock= %u \n", prop.maxThreadsPerBlock);
            printf("maxGridSize= %i \n", prop.maxGridSize[0]);
            printf("sharedMemPerBlock= %lu \n", prop.sharedMemPerBlock);
            printf("deviceOverlap= %i \n", prop.deviceOverlap);
            printf("multiProcessorCount= %i \n", prop.multiProcessorCount);
        }

    size_t size = signal_length_samples * sizeof(GPU_Complex);

    // Doppler-free signal (internal GPU memory)
    cudaMalloc((void **)&d_sig_doppler_wiped, size);
    cudaMemset(d_sig_doppler_wiped, 0, size);

    // Local code GPU memory
    cudaMalloc((void **)&d_local_codes_in, sizeof(std::complex<float>) * code_length_chips);
    cudaMemset(d_local_codes_in, 0, sizeof(std::complex<float>) * code_length_chips);

    d_code_length_chips = code_length_chips;

    // Correlator chip shifts
    cudaMalloc((void **)&d_shifts_chips, sizeof(float) * n_correlators);
    cudaMemset(d_shifts_chips, 0, sizeof(float) * n_correlators);

    threadsPerBlock = 64;
    blocksPerGrid = 128;

    cudaStreamCreate(&stream1);
    return true;
}


bool cuda_multicorrelator::set_local_code_and_taps(
    int code_length_chips,
    const std::complex<float> *local_codes_in,
    float *shifts_chips,
    int n_correlators)
{
    cudaSetDevice(selected_gps_device);
    cudaMemcpyAsync(d_local_codes_in, local_codes_in, sizeof(GPU_Complex) * code_length_chips, cudaMemcpyHostToDevice, stream1);
    d_code_length_chips = code_length_chips;
    cudaMemcpyAsync(d_shifts_chips, shifts_chips, sizeof(float) * n_correlators,
        cudaMemcpyHostToDevice, stream1);
    return true;
}


bool cuda_multicorrelator::set_input_output_vectors(
    std::complex<float> *corr_out,
    std::complex<float> *sig_in)
{
    cudaSetDevice(selected_gps_device);
    d_sig_in_cpu = sig_in;
    d_corr_out_cpu = corr_out;

    cudaError_t code;
    code = cudaHostGetDevicePointer((void **)&d_sig_in, (void *)sig_in, 0);
    code = cudaHostGetDevicePointer((void **)&d_corr_out, (void *)corr_out, 0);
    if (code != cudaSuccess)
        {
            printf("cuda cudaHostGetDevicePointer error \r\n");
        }
    return true;
}


cuda_multicorrelator::cuda_multicorrelator()
{
    d_sig_in = NULL;
    d_nco_in = NULL;
    d_sig_doppler_wiped = NULL;
    d_local_codes_in = NULL;
    d_shifts_samples = NULL;
    d_shifts_chips = NULL;
    d_corr_out = NULL;
    threadsPerBlock = 0;
    blocksPerGrid = 0;
    d_code_length_chips = 0;
}


bool cuda_multicorrelator::free_cuda()
{
    if (d_sig_in != NULL) cudaFree(d_sig_in);
    if (d_nco_in != NULL) cudaFree(d_nco_in);
    if (d_sig_doppler_wiped != NULL) cudaFree(d_sig_doppler_wiped);
    if (d_local_codes_in != NULL) cudaFree(d_local_codes_in);
    if (d_corr_out != NULL) cudaFree(d_corr_out);
    if (d_shifts_samples != NULL) cudaFree(d_shifts_samples);
    if (d_shifts_chips != NULL) cudaFree(d_shifts_chips);
    cudaDeviceReset();
    return true;
}
