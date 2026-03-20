/*!
 * \file pcps_acquisition_cuda_kernels.cu
 * \brief cuFFT-based PCPS acquisition CUDA kernels for batched Doppler search
 *
 * Replaces the serial Doppler loop in pcps_acquisition with parallel GPU
 * execution via cuFFT batched transforms and element-wise CUDA kernels.
 * Targets Jetson Orin Nano (Ampere SM 8.7, unified memory).
 *
 * Derived from gnss-station cuFFT acquisition prototype.
 *
 * \author gnss-station project, 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcps_acquisition_cuda_kernels.h"
#include <cstdio>
#include <cstring>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void conjugate_kernel(cufftComplex* data, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        {
            data[i].y = -data[i].y;
        }
}


__global__ void doppler_wipeoff_kernel(
    const cufftComplex* __restrict__ signal,
    cufftComplex* __restrict__ output,
    uint32_t fft_size,
    uint32_t num_doppler_bins,
    float doppler_max_hz,
    float doppler_step_hz,
    float doppler_center_hz,
    float fs_hz)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;

    if (n >= fft_size || b >= num_doppler_bins) return;

    const float f_doppler = -doppler_max_hz + doppler_step_hz * static_cast<float>(b) + doppler_center_hz;
    const float phase = -2.0f * M_PI * f_doppler * static_cast<float>(n) / fs_hz;

    float cos_val, sin_val;
    __sincosf(phase, &sin_val, &cos_val);

    const cufftComplex s = signal[n];
    const uint32_t idx = b * fft_size + n;
    output[idx].x = s.x * cos_val - s.y * sin_val;
    output[idx].y = s.x * sin_val + s.y * cos_val;
}


__global__ void spectral_multiply_kernel(
    const cufftComplex* __restrict__ fft_signal,
    const cufftComplex* __restrict__ code_fft_conj,
    cufftComplex* __restrict__ output,
    uint32_t fft_size,
    uint32_t num_doppler_bins)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;

    if (n >= fft_size || b >= num_doppler_bins) return;

    const uint32_t idx = b * fft_size + n;
    const cufftComplex a = fft_signal[idx];
    const cufftComplex c = code_fft_conj[n];

    output[idx].x = a.x * c.x - a.y * c.y;
    output[idx].y = a.x * c.y + a.y * c.x;
}


__global__ void magnitude_squared_kernel(
    const cufftComplex* __restrict__ corr_output,
    float* __restrict__ magnitude_grid,
    uint32_t fft_size,
    uint32_t num_doppler_bins)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;

    if (n >= fft_size || b >= num_doppler_bins) return;

    const uint32_t idx = b * fft_size + n;
    const float re = corr_output[idx].x;
    const float im = corr_output[idx].y;
    magnitude_grid[idx] = re * re + im * im;
}


__global__ void magnitude_squared_accumulate_kernel(
    const cufftComplex* __restrict__ corr_output,
    float* __restrict__ magnitude_grid,
    uint32_t fft_size,
    uint32_t num_doppler_bins)
{
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;

    if (n >= fft_size || b >= num_doppler_bins) return;

    const uint32_t idx = b * fft_size + n;
    const float re = corr_output[idx].x;
    const float im = corr_output[idx].y;
    magnitude_grid[idx] += re * re + im * im;
}


// ============================================================================
// Host-side functions
// ============================================================================

void cuacq_init(CuAcqState* state,
    uint32_t fft_size,
    uint32_t num_doppler_bins,
    float doppler_max_hz,
    float doppler_step_hz,
    float fs_hz)
{
    state->fft_size = fft_size;
    state->num_doppler_bins = num_doppler_bins;
    state->doppler_max_hz = doppler_max_hz;
    state->doppler_step_hz = doppler_step_hz;
    state->fs_hz = fs_hz;
    state->doppler_center_hz = 0.0f;

    const size_t batch_size = static_cast<size_t>(num_doppler_bins) * fft_size;

    // Unified memory for Jetson (no PCIe copy overhead)
    cudaMallocManaged(&state->d_signal, fft_size * sizeof(cufftComplex));
    cudaMallocManaged(&state->d_code_fft_conj, fft_size * sizeof(cufftComplex));
    cudaMallocManaged(&state->d_doppler_removed, batch_size * sizeof(cufftComplex));
    cudaMallocManaged(&state->d_spectral_product, batch_size * sizeof(cufftComplex));
    cudaMallocManaged(&state->d_corr_ifft_out, batch_size * sizeof(cufftComplex));
    cudaMallocManaged(&state->d_magnitude_grid, batch_size * sizeof(float));

    // Peak result scalars
    cudaMallocManaged(&state->d_peak_value, sizeof(float));
    cudaMallocManaged(&state->d_peak_index, sizeof(uint32_t));

    // Batched cuFFT plans
    cufftPlan1d(&state->fft_plan_fwd, fft_size, CUFFT_C2C, num_doppler_bins);
    cufftPlan1d(&state->fft_plan_inv, fft_size, CUFFT_C2C, num_doppler_bins);

    // Single-transform plan for code FFT
    cufftPlan1d(&state->fft_plan_single, fft_size, CUFFT_C2C, 1);
}


void cuacq_set_local_code(CuAcqState* state, const std::complex<float>* code_samples)
{
    // Copy code into device signal buffer
    memcpy(state->d_signal, code_samples, state->fft_size * sizeof(cufftComplex));

    // Forward FFT of the local code
    cufftExecC2C(state->fft_plan_single, state->d_signal, state->d_code_fft_conj, CUFFT_FORWARD);

    // Conjugate the FFT result
    uint32_t tpb = 256;
    uint32_t blocks = (state->fft_size + tpb - 1) / tpb;
    conjugate_kernel<<<blocks, tpb>>>(state->d_code_fft_conj, state->fft_size);

    cudaDeviceSynchronize();
}


void cuacq_acquisition_core(CuAcqState* state,
    const std::complex<float>* input_signal,
    bool accumulate,
    uint32_t effective_fft_size)
{
    const uint32_t N = state->fft_size;
    const uint32_t B = state->num_doppler_bins;

    // Copy input signal to unified memory
    memcpy(state->d_signal, input_signal, N * sizeof(cufftComplex));

    // Kernel launch configuration
    const uint32_t threads_per_block = 256;
    const dim3 block(threads_per_block, 1);
    const dim3 grid((N + threads_per_block - 1) / threads_per_block, B);

    // Step 1: Doppler wipeoff (all bins, one kernel launch)
    doppler_wipeoff_kernel<<<grid, block>>>(
        state->d_signal,
        state->d_doppler_removed,
        N, B,
        state->doppler_max_hz,
        state->doppler_step_hz,
        state->doppler_center_hz,
        state->fs_hz);

    // Step 2: Batched forward FFT
    cufftExecC2C(state->fft_plan_fwd,
        state->d_doppler_removed,
        state->d_doppler_removed,
        CUFFT_FORWARD);

    // Step 3: Spectral multiply with code FFT conjugate
    spectral_multiply_kernel<<<grid, block>>>(
        state->d_doppler_removed,
        state->d_code_fft_conj,
        state->d_spectral_product,
        N, B);

    // Step 4: Batched inverse FFT
    cufftExecC2C(state->fft_plan_inv,
        state->d_spectral_product,
        state->d_corr_ifft_out,
        CUFFT_INVERSE);

    // Step 5: Magnitude squared (with optional accumulation for non-coherent integration)
    if (accumulate)
        {
            magnitude_squared_accumulate_kernel<<<grid, block>>>(
                state->d_corr_ifft_out,
                state->d_magnitude_grid,
                N, B);
        }
    else
        {
            magnitude_squared_kernel<<<grid, block>>>(
                state->d_corr_ifft_out,
                state->d_magnitude_grid,
                N, B);
        }

    cudaDeviceSynchronize();
}


void cuacq_find_peak(CuAcqState* state,
    uint32_t effective_fft_size,
    uint32_t* out_code_phase,
    uint32_t* out_doppler_bin,
    float* out_peak_magnitude)
{
    // CPU scan — the grid is small enough (~80K floats for GPS L1 C/A) that
    // this is negligible compared to FFT time. For wideband modes (56 MSPS)
    // this could be moved to a GPU reduction kernel if needed.
    const uint32_t N = state->fft_size;
    const uint32_t B = state->num_doppler_bins;

    float best_mag = -1.0f;
    uint32_t best_bin = 0;
    uint32_t best_phase = 0;

    for (uint32_t b = 0; b < B; b++)
        {
            for (uint32_t n = 0; n < effective_fft_size; n++)
                {
                    float val = state->d_magnitude_grid[b * N + n];
                    if (val > best_mag)
                        {
                            best_mag = val;
                            best_bin = b;
                            best_phase = n;
                        }
                }
        }

    *out_doppler_bin = best_bin;
    *out_code_phase = best_phase;
    *out_peak_magnitude = best_mag;
}


float cuacq_compute_input_power(CuAcqState* state, uint32_t effective_fft_size)
{
    const uint32_t N = state->fft_size;
    const uint32_t B = state->num_doppler_bins;
    double sum = 0.0;
    uint32_t total = B * effective_fft_size;
    for (uint32_t b = 0; b < B; b++)
        {
            for (uint32_t n = 0; n < effective_fft_size; n++)
                {
                    sum += static_cast<double>(state->d_magnitude_grid[b * N + n]);
                }
        }
    return static_cast<float>(sum / total);
}


float cuacq_second_peak(CuAcqState* state,
    uint32_t peak_doppler_bin,
    uint32_t peak_code_phase,
    uint32_t effective_fft_size,
    uint32_t samples_per_chip)
{
    const uint32_t N = state->fft_size;

    // Exclude a window of +/- samples_per_chip around the peak in the same
    // Doppler row (same approach as pcps_acquisition::first_vs_second_peak_statistic)
    int32_t exclude_start = static_cast<int32_t>(peak_code_phase) - static_cast<int32_t>(samples_per_chip);
    int32_t exclude_end = static_cast<int32_t>(peak_code_phase) + static_cast<int32_t>(samples_per_chip);

    if (exclude_start < 0) exclude_start += effective_fft_size;
    if (exclude_end >= static_cast<int32_t>(effective_fft_size)) exclude_end -= effective_fft_size;

    float second_peak = 0.0f;
    const float* row = &state->d_magnitude_grid[peak_doppler_bin * N];

    for (uint32_t n = 0; n < effective_fft_size; n++)
        {
            // Check if sample is in exclusion zone (handles wrap-around)
            bool excluded = false;
            if (exclude_start <= exclude_end)
                {
                    excluded = (static_cast<int32_t>(n) >= exclude_start &&
                        static_cast<int32_t>(n) <= exclude_end);
                }
            else
                {
                    excluded = (static_cast<int32_t>(n) >= exclude_start ||
                        static_cast<int32_t>(n) <= exclude_end);
                }
            if (!excluded && row[n] > second_peak)
                {
                    second_peak = row[n];
                }
        }

    return second_peak;
}


void cuacq_update_doppler_params(CuAcqState* state,
    uint32_t num_doppler_bins,
    float doppler_max_hz,
    float doppler_step_hz,
    float doppler_center_hz)
{
    bool need_replan = (state->num_doppler_bins != num_doppler_bins);

    state->num_doppler_bins = num_doppler_bins;
    state->doppler_max_hz = doppler_max_hz;
    state->doppler_step_hz = doppler_step_hz;
    state->doppler_center_hz = doppler_center_hz;

    if (need_replan)
        {
            // Reallocate batch buffers and recreate plans
            const size_t batch_size = static_cast<size_t>(num_doppler_bins) * state->fft_size;

            cudaFree(state->d_doppler_removed);
            cudaFree(state->d_spectral_product);
            cudaFree(state->d_corr_ifft_out);
            cudaFree(state->d_magnitude_grid);

            cudaMallocManaged(&state->d_doppler_removed, batch_size * sizeof(cufftComplex));
            cudaMallocManaged(&state->d_spectral_product, batch_size * sizeof(cufftComplex));
            cudaMallocManaged(&state->d_corr_ifft_out, batch_size * sizeof(cufftComplex));
            cudaMallocManaged(&state->d_magnitude_grid, batch_size * sizeof(float));

            cufftDestroy(state->fft_plan_fwd);
            cufftDestroy(state->fft_plan_inv);
            cufftPlan1d(&state->fft_plan_fwd, state->fft_size, CUFFT_C2C, num_doppler_bins);
            cufftPlan1d(&state->fft_plan_inv, state->fft_size, CUFFT_C2C, num_doppler_bins);
        }
}


void cuacq_destroy(CuAcqState* state)
{
    cufftDestroy(state->fft_plan_fwd);
    cufftDestroy(state->fft_plan_inv);
    cufftDestroy(state->fft_plan_single);

    cudaFree(state->d_signal);
    cudaFree(state->d_code_fft_conj);
    cudaFree(state->d_doppler_removed);
    cudaFree(state->d_spectral_product);
    cudaFree(state->d_corr_ifft_out);
    cudaFree(state->d_magnitude_grid);
    cudaFree(state->d_peak_value);
    cudaFree(state->d_peak_index);
}
