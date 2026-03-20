/*!
 * \file pcps_acquisition_cuda.h
 * \brief cuFFT-based Parallel Code Phase Search Acquisition block
 *
 * Drop-in replacement for pcps_acquisition that uses cuFFT batched transforms
 * to execute all Doppler bins in parallel on GPU. Same GNU Radio interface,
 * same messaging protocol, same config parameters — just a different engine.
 *
 * Target: Jetson Orin Nano (Ampere SM 8.7, CUDA 12.6, unified memory).
 *
 * \author gnss-station project, 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef GNSS_SDR_PCPS_ACQUISITION_CUDA_H
#define GNSS_SDR_PCPS_ACQUISITION_CUDA_H

#include "acq_conf.h"
#include "channel_fsm.h"
#include "pcps_acquisition_cuda_kernels.h"
#include <gnuradio/block.h>
#include <gnuradio/gr_complex.h>
#include <gnuradio/thread/thread.h>
#include <gnuradio/types.h>
#include <volk/volk_complex.h>
#include <volk_gnsssdr/volk_gnsssdr_alloc.h>
#include <complex>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <utility>

#if HAS_STD_SPAN
#include <span>
namespace own = std;
#else
#include <gsl/gsl-lite.hpp>
namespace own = gsl;
#endif

/** \addtogroup Acquisition
 * \{ */
/** \addtogroup Acq_gnuradio_blocks acquisition_gr_blocks
 * \{ */


class Gnss_Synchro;
class pcps_acquisition_cuda;

using pcps_acquisition_cuda_sptr = gnss_shared_ptr<pcps_acquisition_cuda>;

pcps_acquisition_cuda_sptr pcps_make_acquisition_cuda(const Acq_Conf& conf_);

/*!
 * \brief cuFFT-accelerated Parallel Code Phase Search Acquisition.
 *
 * Replaces the serial Doppler loop in pcps_acquisition with batched cuFFT
 * transforms. The interface is identical to pcps_acquisition so adapters
 * can use it as a drop-in replacement.
 */
class pcps_acquisition_cuda : public gr::block
{
public:
    ~pcps_acquisition_cuda() override;

    void init();

    inline void set_gnss_synchro(Gnss_Synchro* p_gnss_synchro)
    {
        gr::thread::scoped_lock lock(d_setlock);
        d_gnss_synchro = p_gnss_synchro;
    }

    void set_local_code(std::complex<float>* code);

    void set_state(int32_t state);

    void set_resampler_latency(uint32_t latency_samples);

    inline uint32_t mag() const
    {
        return d_mag;
    }

    inline void set_active(bool active)
    {
        gr::thread::scoped_lock lock(d_setlock);
        d_active = active;
    }

    inline void set_channel(uint32_t channel)
    {
        d_channel = channel;
    }

    inline void set_channel_fsm(std::weak_ptr<ChannelFsm> channel_fsm)
    {
        d_channel_fsm = std::move(channel_fsm);
    }

    inline void set_threshold(float threshold)
    {
        gr::thread::scoped_lock lock(d_setlock);
        d_threshold = threshold;
    }

    inline void set_doppler_max(uint32_t doppler_max)
    {
        gr::thread::scoped_lock lock(d_setlock);
        d_acq_parameters.doppler_max = doppler_max;
    }

    inline void set_doppler_step(uint32_t doppler_step)
    {
        gr::thread::scoped_lock lock(d_setlock);
        d_doppler_step = doppler_step;
    }

    void set_doppler_center(int32_t doppler_center);

    int general_work(int noutput_items, gr_vector_int& ninput_items,
        gr_vector_const_void_star& input_items,
        gr_vector_void_star& output_items) override;

private:
    friend pcps_acquisition_cuda_sptr pcps_make_acquisition_cuda(const Acq_Conf& conf_);
    explicit pcps_acquisition_cuda(const Acq_Conf& conf_);

    void acquisition_core(uint64_t samp_count);
    void send_negative_acquisition();
    void send_positive_acquisition();
    bool is_fdma();
    bool start() override;
    void calculate_threshold(void);

    // CUDA state
    CuAcqState d_cuda_state;
    bool d_cuda_initialized;
    volk_gnsssdr::vector<std::complex<float>> d_cached_code_buf;  // deferred until CUDA init

    // Data buffers (host side, for GNU Radio interface)
    volk_gnsssdr::vector<std::complex<float>> d_input_signal;
    volk_gnsssdr::vector<std::complex<float>> d_data_buffer;
    volk_gnsssdr::vector<lv_16sc_t> d_data_buffer_sc;

    std::weak_ptr<ChannelFsm> d_channel_fsm;

    Acq_Conf d_acq_parameters;
    Gnss_Synchro* d_gnss_synchro;

    std::queue<Gnss_Synchro> d_monitor_queue;
    std::string d_dump_filename;

    int64_t d_dump_number;
    uint64_t d_sample_counter;

    float d_threshold;
    float d_mag;
    float d_input_power;
    float d_test_statistics;
    float d_doppler_center_step_two;

    int32_t d_state;
    int32_t d_positive_acq;
    int32_t d_doppler_center;
    int32_t d_doppler_bias;
    uint32_t d_channel;
    uint32_t d_samplesPerChip;
    uint32_t d_doppler_step;
    uint32_t d_num_noncoherent_integrations_counter;
    uint32_t d_fft_size;
    uint32_t d_consumed_samples;
    uint32_t d_num_doppler_bins;
    uint32_t d_num_doppler_bins_step2;
    uint32_t d_dump_channel;
    uint32_t d_buffer_count;

    bool d_active;
    bool d_worker_active;
    bool d_cshort;
    bool d_step_two;
    bool d_use_CFAR_algorithm_flag;
    bool d_dump;
};


/** \} */
/** \} */
#endif  // GNSS_SDR_PCPS_ACQUISITION_CUDA_H
