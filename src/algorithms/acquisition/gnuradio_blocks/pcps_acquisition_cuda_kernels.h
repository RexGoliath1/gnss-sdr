/*!
 * \file pcps_acquisition_cuda_kernels.h
 * \brief Header for cuFFT-based PCPS acquisition CUDA kernels
 *
 * \author gnss-station project, 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef GNSS_SDR_PCPS_ACQUISITION_CUDA_KERNELS_H
#define GNSS_SDR_PCPS_ACQUISITION_CUDA_KERNELS_H

#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <cstdint>

/*!
 * \brief Holds all GPU-side state for one cuFFT acquisition channel.
 *
 * Memory strategy: cudaMallocManaged (unified memory) for Jetson Orin.
 * On discrete GPUs, replace with cudaMalloc + cudaMemcpyAsync.
 */
struct CuAcqState
{
    // Configuration
    uint32_t fft_size;
    uint32_t num_doppler_bins;
    float doppler_max_hz;
    float doppler_step_hz;
    float doppler_center_hz;
    float fs_hz;

    // Device buffers (all cudaMallocManaged)
    cufftComplex* d_signal;
    cufftComplex* d_code_fft_conj;
    cufftComplex* d_doppler_removed;
    cufftComplex* d_spectral_product;
    cufftComplex* d_corr_ifft_out;
    float* d_magnitude_grid;

    // Peak result scalars
    float* d_peak_value;
    uint32_t* d_peak_index;

    // cuFFT plans
    cufftHandle fft_plan_fwd;
    cufftHandle fft_plan_inv;
    cufftHandle fft_plan_single;

    // Per-channel CUDA stream (avoids default-stream serialization conflicts
    // when multiple acquisition channels run concurrently)
    cudaStream_t stream;
};


/*!
 * \brief Global CUDA mutex lock/unlock — serializes all GPU operations and
 * CPU reads of unified memory across acquisition channels. Caller must bracket
 * the entire GPU work + readback sequence with cuacq_lock()/cuacq_unlock().
 */
void cuacq_lock();
void cuacq_unlock();

/*!
 * \brief Initialize cuFFT acquisition state — called once per channel at startup.
 * Caller must hold cuacq_lock().
 */
void cuacq_init(CuAcqState* state,
    uint32_t fft_size,
    uint32_t num_doppler_bins,
    float doppler_max_hz,
    float doppler_step_hz,
    float fs_hz);

/*!
 * \brief Set the local PRN code — called once per satellite search.
 * Computes FFT of code and stores conjugate for correlation.
 */
void cuacq_set_local_code(CuAcqState* state, const std::complex<float>* code_samples);

/*!
 * \brief Core acquisition — runs the batched cuFFT PCPS pipeline.
 * \param accumulate If true, adds to existing magnitude grid (non-coherent integration).
 * \param effective_fft_size Number of samples to use (may be fft_size/2 for bit transition).
 */
void cuacq_acquisition_core(CuAcqState* state,
    const std::complex<float>* input_signal,
    bool accumulate,
    uint32_t effective_fft_size);

/*!
 * \brief Find the peak in the magnitude grid.
 */
void cuacq_find_peak(CuAcqState* state,
    uint32_t effective_fft_size,
    uint32_t* out_code_phase,
    uint32_t* out_doppler_bin,
    float* out_peak_magnitude);

/*!
 * \brief Compute mean magnitude for CFAR test statistic.
 */
float cuacq_compute_input_power(CuAcqState* state, uint32_t effective_fft_size);

/*!
 * \brief Find second peak (excluding window around first peak) for first-vs-second test.
 */
float cuacq_second_peak(CuAcqState* state,
    uint32_t peak_doppler_bin,
    uint32_t peak_code_phase,
    uint32_t effective_fft_size,
    uint32_t samples_per_chip);

/*!
 * \brief Update Doppler search parameters (and reallocate if num_bins changed).
 */
void cuacq_update_doppler_params(CuAcqState* state,
    uint32_t num_doppler_bins,
    float doppler_max_hz,
    float doppler_step_hz,
    float doppler_center_hz);

/*!
 * \brief Free all GPU memory and destroy cuFFT plans.
 */
void cuacq_destroy(CuAcqState* state);

#endif  // GNSS_SDR_PCPS_ACQUISITION_CUDA_KERNELS_H
