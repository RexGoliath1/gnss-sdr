/*!
 * \file pcps_acquisition_cuda.cc
 * \brief cuFFT-based Parallel Code Phase Search Acquisition block
 *
 * \author gnss-station project, 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pcps_acquisition_cuda.h"
#include "GLONASS_L1_L2_CA.h"
#include "MATH_CONSTANTS.h"
#include "gnss_frequencies.h"
#include "gnss_synchro.h"
#include <boost/math/special_functions/gamma.hpp>
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>
#include <pmt/pmt_sugar.h>
#include <volk/volk.h>
#include <volk_gnsssdr/volk_gnsssdr.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#if USE_GLOG_AND_GFLAGS
#include <glog/logging.h>
#else
#include <absl/log/log.h>
#endif


pcps_acquisition_cuda_sptr pcps_make_acquisition_cuda(const Acq_Conf& conf_)
{
    return pcps_acquisition_cuda_sptr(new pcps_acquisition_cuda(conf_));
}


pcps_acquisition_cuda::pcps_acquisition_cuda(const Acq_Conf& conf_)
    : gr::block("pcps_acquisition_cuda",
          gr::io_signature::make(1, 1, conf_.it_size),
          gr::io_signature::make(0, 1, sizeof(Gnss_Synchro))),
      d_cuda_initialized(false),
      d_acq_parameters(conf_),
      d_gnss_synchro(nullptr),
      d_dump_filename(conf_.dump_filename),
      d_dump_number(0LL),
      d_sample_counter(0ULL),
      d_threshold(0.0),
      d_mag(0),
      d_input_power(0.0),
      d_test_statistics(0.0),
      d_doppler_center_step_two(0.0),
      d_state(0),
      d_positive_acq(0),
      d_doppler_center(0U),
      d_doppler_bias(0),
      d_channel(0U),
      d_samplesPerChip(conf_.samples_per_chip),
      d_doppler_step(conf_.doppler_step),
      d_num_noncoherent_integrations_counter(0U),
      d_consumed_samples(conf_.sampled_ms * conf_.samples_per_ms * (conf_.bit_transition_flag ? 2.0 : 1.0)),
      d_num_doppler_bins(0U),
      d_num_doppler_bins_step2(conf_.num_doppler_bins_step2),
      d_dump_channel(conf_.dump_channel),
      d_buffer_count(0U),
      d_active(false),
      d_worker_active(false),
      d_step_two(false),
      d_use_CFAR_algorithm_flag(conf_.use_CFAR_algorithm_flag),
      d_dump(conf_.dump)
{
    this->message_port_register_out(pmt::mp("events"));

    if (d_acq_parameters.sampled_ms == d_acq_parameters.ms_per_code)
        {
            d_fft_size = d_consumed_samples;
        }
    else
        {
            d_fft_size = d_consumed_samples * 2;
        }

    d_input_signal = volk_gnsssdr::vector<std::complex<float>>(d_fft_size);

    if (conf_.it_size == sizeof(gr_complex))
        {
            d_cshort = false;
        }
    else
        {
            d_cshort = true;
        }

    d_data_buffer = volk_gnsssdr::vector<std::complex<float>>(d_consumed_samples);
    if (d_cshort)
        {
            d_data_buffer_sc = volk_gnsssdr::vector<lv_16sc_t>(d_consumed_samples);
        }

    // Zero-initialize CUDA state (will be properly initialized in init())
    std::memset(&d_cuda_state, 0, sizeof(CuAcqState));
}


pcps_acquisition_cuda::~pcps_acquisition_cuda()
{
    if (d_cuda_initialized)
        {
            cuacq_destroy(&d_cuda_state);
        }
}


void pcps_acquisition_cuda::set_resampler_latency(uint32_t latency_samples)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_acq_parameters.resampler_latency_samples = latency_samples;
}


void pcps_acquisition_cuda::set_local_code(std::complex<float>* code)
{
    if (is_fdma())
        {
            // For FDMA, update Doppler grid to account for frequency offset
        }

    gr::thread::scoped_lock lock(d_setlock);

    // Prepare code buffer (same logic as pcps_acquisition)
    volk_gnsssdr::vector<std::complex<float>> code_buf(d_fft_size, std::complex<float>(0.0f, 0.0f));

    if (d_acq_parameters.bit_transition_flag)
        {
            const int32_t offset = d_fft_size / 2;
            std::copy(code, code + offset, code_buf.data() + offset);
        }
    else
        {
            if (d_acq_parameters.sampled_ms == d_acq_parameters.ms_per_code)
                {
                    std::copy(code, code + d_consumed_samples, code_buf.data());
                }
            else
                {
                    std::copy(code, code + d_consumed_samples, code_buf.data() + d_consumed_samples);
                }
        }

    // Set local code on GPU (computes FFT + conjugate)
    // Guard: CUDA state may not be initialized yet (init() called after set_local_code).
    // Cache the code buffer and replay when init() runs.
    d_cached_code_buf = code_buf;
    if (d_cuda_initialized)
        {
            cuacq_set_local_code(&d_cuda_state, d_cached_code_buf.data());
        }
}


bool pcps_acquisition_cuda::is_fdma()
{
    d_doppler_bias = 0;
    if (strcmp(d_gnss_synchro->Signal, "1G") == 0)
        {
            d_doppler_bias = static_cast<int32_t>(DFRQ1_GLO * GLONASS_PRN.at(d_gnss_synchro->PRN));
            DLOG(INFO) << "Trying to acquire SV PRN " << d_gnss_synchro->PRN
                       << " with freq " << d_doppler_bias
                       << " in Glonass Channel " << GLONASS_PRN.at(d_gnss_synchro->PRN);
            return true;
        }
    if (strcmp(d_gnss_synchro->Signal, "2G") == 0)
        {
            d_doppler_bias += static_cast<int32_t>(DFRQ2_GLO * GLONASS_PRN.at(d_gnss_synchro->PRN));
            DLOG(INFO) << "Trying to acquire SV PRN " << d_gnss_synchro->PRN
                       << " with freq " << d_doppler_bias
                       << " in Glonass Channel " << GLONASS_PRN.at(d_gnss_synchro->PRN);
            return true;
        }
    return false;
}


void pcps_acquisition_cuda::init()
{
    d_gnss_synchro->Flag_valid_acquisition = false;
    d_gnss_synchro->Flag_valid_symbol_output = false;
    d_gnss_synchro->Flag_valid_pseudorange = false;
    d_gnss_synchro->Flag_valid_word = false;
    d_gnss_synchro->Acq_doppler_step = 0U;
    d_gnss_synchro->Acq_delay_samples = 0.0;
    d_gnss_synchro->Acq_doppler_hz = 0.0;
    d_gnss_synchro->Acq_samplestamp_samples = 0ULL;

    d_mag = 0.0;
    d_input_power = 0.0;

    d_num_doppler_bins = static_cast<uint32_t>(std::ceil(static_cast<double>(static_cast<int32_t>(d_acq_parameters.doppler_max) - (-static_cast<int32_t>(d_acq_parameters.doppler_max))) / static_cast<double>(d_doppler_step)));

    // Initialize CUDA state
    float fs = d_acq_parameters.use_automatic_resampler
                   ? static_cast<float>(d_acq_parameters.resampled_fs)
                   : static_cast<float>(d_acq_parameters.fs_in);

    // CUDA init is deferred to acquisition_core() because main.cc calls
    // cudaDeviceReset() AFTER flowgraph construction, which invalidates
    // all prior cudaMallocManaged allocations. Store params for lazy init.
    d_cuda_fs = fs;

    if (d_cuda_initialized)
        {
            // Update parameters if already initialized
            cuacq_update_doppler_params(&d_cuda_state,
                d_num_doppler_bins,
                static_cast<float>(d_acq_parameters.doppler_max),
                static_cast<float>(d_doppler_step),
                static_cast<float>(d_doppler_center + d_doppler_bias));
        }

    calculate_threshold();
}


void pcps_acquisition_cuda::set_state(int32_t state)
{
    gr::thread::scoped_lock lock(d_setlock);
    d_state = state;
    if (d_state == 1)
        {
            d_gnss_synchro->Acq_delay_samples = 0.0;
            d_gnss_synchro->Acq_doppler_hz = 0.0;
            d_gnss_synchro->Acq_samplestamp_samples = 0ULL;
            d_gnss_synchro->Acq_doppler_step = 0U;
            d_mag = 0.0;
            d_buffer_count = 0U;
            d_num_noncoherent_integrations_counter = 0U;
            d_positive_acq = 0;
            d_active = true;
        }
    else if (d_state == 0)
        {
            // Nothing
        }
    else
        {
            LOG(ERROR) << "State can only be set to 0 or 1";
        }
}


void pcps_acquisition_cuda::set_doppler_center(int32_t doppler_center)
{
    gr::thread::scoped_lock lock(d_setlock);
    if (doppler_center != d_doppler_center)
        {
            DLOG(INFO) << " Doppler assistance for Channel: " << d_channel
                       << " => Doppler: " << doppler_center << "[Hz]";
            d_doppler_center = doppler_center;
            if (d_cuda_initialized)
                {
                    d_cuda_state.doppler_center_hz = static_cast<float>(d_doppler_center + d_doppler_bias);
                }
        }
}


void pcps_acquisition_cuda::calculate_threshold()
{
    const float pfa = (d_step_two ? d_acq_parameters.pfa2 : d_acq_parameters.pfa);

    if (pfa <= 0.0)
        {
            return;
        }

    const auto effective_fft_size = static_cast<int>(d_acq_parameters.bit_transition_flag ? (d_fft_size / 2) : d_fft_size);
    const int num_doppler_bins = (d_step_two ? d_num_doppler_bins_step2 : d_num_doppler_bins);
    const int num_bins = effective_fft_size * num_doppler_bins;

    d_threshold = static_cast<float>(2.0 * boost::math::gamma_p_inv(
        2.0 * (d_acq_parameters.bit_transition_flag ? 1 : d_acq_parameters.max_dwells),
        std::pow(1.0 - pfa, 1.0 / static_cast<float>(num_bins))));
}


void pcps_acquisition_cuda::send_positive_acquisition()
{
    // Declare positive acquisition using a message port
    // 6.1- Declare positive acquisition using a message port
    DLOG(INFO) << "positive acquisition"
               << ", satellite " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << ", sample_stamp " << d_gnss_synchro->Acq_samplestamp_samples
               << ", test statistics value " << d_test_statistics
               << ", test statistics threshold " << d_threshold
               << ", code phase " << d_gnss_synchro->Acq_delay_samples
               << ", doppler " << d_gnss_synchro->Acq_doppler_hz
               << ", magnitude " << d_mag
               << ", input signal power " << d_input_power;
    d_positive_acq = 1;

    if (!d_channel_fsm.expired())
        {
            // The channel FSM is set, so call the event
            d_channel_fsm.lock()->Event_valid_acquisition();
        }

    if (d_acq_parameters.enable_monitor_output)
        {
            d_gnss_synchro->Flag_valid_acquisition = true;
            d_monitor_queue.push(*d_gnss_synchro);
        }
}


void pcps_acquisition_cuda::send_negative_acquisition()
{
    // 6.2- Declare negative acquisition using a message port
    DLOG(INFO) << "negative acquisition"
               << ", satellite " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << ", sample_stamp " << d_gnss_synchro->Acq_samplestamp_samples
               << ", test statistics value " << d_test_statistics
               << ", test statistics threshold " << d_threshold
               << ", code phase " << d_gnss_synchro->Acq_delay_samples
               << ", doppler " << d_gnss_synchro->Acq_doppler_hz
               << ", magnitude " << d_mag
               << ", input signal power " << d_input_power;

    d_positive_acq = 0;
    this->message_port_pub(pmt::mp("events"), pmt::from_long(2));

    if (d_acq_parameters.enable_monitor_output)
        {
            d_gnss_synchro->Flag_valid_acquisition = false;
            d_monitor_queue.push(*d_gnss_synchro);
        }
}


void pcps_acquisition_cuda::acquisition_core(uint64_t samp_count)
{
    gr::thread::scoped_lock lk(d_setlock);

    // Lazy CUDA init: deferred from init() because main.cc calls cudaDeviceReset()
    // after flowgraph construction, which invalidates all prior CUDA allocations.
    if (!d_cuda_initialized)
        {
            cuacq_init(&d_cuda_state,
                d_fft_size,
                d_num_doppler_bins,
                static_cast<float>(d_acq_parameters.doppler_max),
                static_cast<float>(d_doppler_step),
                d_cuda_fs);
            d_cuda_state.doppler_center_hz = static_cast<float>(d_doppler_center + d_doppler_bias);
            d_cuda_initialized = true;
            if (!d_cached_code_buf.empty())
                {
                    cuacq_set_local_code(&d_cuda_state, d_cached_code_buf.data());
                }
        }

    int32_t doppler = 0;
    uint32_t indext = 0U;
    const int32_t effective_fft_size = (d_acq_parameters.bit_transition_flag ? d_fft_size / 2 : d_fft_size);

    if (d_cshort)
        {
            volk_gnsssdr_16ic_convert_32fc(d_data_buffer.data(), d_data_buffer_sc.data(), d_consumed_samples);
        }
    std::copy(d_data_buffer.data(), d_data_buffer.data() + d_consumed_samples, d_input_signal.data());
    if (d_fft_size > d_consumed_samples)
        {
            for (uint32_t i = d_consumed_samples; i < d_fft_size; i++)
                {
                    d_input_signal[i] = gr_complex(0.0, 0.0);
                }
        }

    d_mag = 0.0;
    d_num_noncoherent_integrations_counter++;

    DLOG(INFO) << "Channel: " << d_channel
               << " , doing CUDA acquisition of satellite: " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << " ,sample stamp: " << samp_count << ", threshold: "
               << d_threshold << ", doppler_max: " << d_acq_parameters.doppler_max
               << ", doppler_step: " << d_doppler_step;

    if (d_acq_parameters.blocking)
        {
            lk.unlock();
        }

    // Run the cuFFT acquisition core — replaces the serial Doppler loop
    if (!d_step_two)
        {
            // Update Doppler center for FDMA if needed
            d_cuda_state.doppler_center_hz = static_cast<float>(d_doppler_center + d_doppler_bias);

            bool accumulate = (d_num_noncoherent_integrations_counter > 1);
            cuacq_acquisition_core(&d_cuda_state,
                d_input_signal.data(),
                accumulate,
                effective_fft_size);

            // Find peak
            uint32_t peak_phase, peak_bin;
            float peak_mag;
            cuacq_find_peak(&d_cuda_state, effective_fft_size,
                &peak_phase, &peak_bin, &peak_mag);

            indext = peak_phase;
            d_mag = peak_mag;

            // Compute test statistic
            if (d_use_CFAR_algorithm_flag)
                {
                    // max_to_input_power_statistic: peak / mean
                    float mean_power = cuacq_compute_input_power(&d_cuda_state, effective_fft_size);
                    if (mean_power > 0.0f)
                        {
                            d_test_statistics = peak_mag / mean_power;
                        }
                    else
                        {
                            d_test_statistics = 0.0f;
                        }
                    d_input_power = mean_power;
                }
            else
                {
                    // first_vs_second_peak_statistic
                    float second_peak = cuacq_second_peak(&d_cuda_state,
                        peak_bin, peak_phase, effective_fft_size, d_samplesPerChip);
                    if (second_peak > 0.0f)
                        {
                            d_test_statistics = peak_mag / second_peak;
                        }
                    else
                        {
                            d_test_statistics = 0.0f;
                        }
                }

            // Convert bin index to Doppler Hz
            doppler = -static_cast<int32_t>(d_acq_parameters.doppler_max) +
                      static_cast<int32_t>(d_doppler_step) * static_cast<int32_t>(peak_bin) +
                      d_doppler_center + d_doppler_bias;

            if (d_acq_parameters.use_automatic_resampler)
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(std::fmod(static_cast<float>(indext), d_acq_parameters.samples_per_code)) * d_acq_parameters.resampler_ratio;
                    d_gnss_synchro->Acq_delay_samples -= static_cast<double>(d_acq_parameters.resampler_latency_samples);
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = rint(static_cast<double>(samp_count) * d_acq_parameters.resampler_ratio);
                    d_gnss_synchro->fs = d_acq_parameters.resampled_fs;
                }
            else
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(std::fmod(static_cast<float>(indext), d_acq_parameters.samples_per_code));
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = samp_count;
                    d_gnss_synchro->fs = d_acq_parameters.fs_in;
                }
        }
    else
        {
            // Step two: fine Doppler search (use same cuFFT engine with updated params)
            float step2_doppler_max = static_cast<float>(d_num_doppler_bins_step2) / 2.0f * d_acq_parameters.doppler_step2;

            cuacq_update_doppler_params(&d_cuda_state,
                d_num_doppler_bins_step2,
                step2_doppler_max,
                d_acq_parameters.doppler_step2,
                d_doppler_center_step_two);

            bool accumulate = (d_num_noncoherent_integrations_counter > 1);
            cuacq_acquisition_core(&d_cuda_state,
                d_input_signal.data(),
                accumulate,
                effective_fft_size);

            uint32_t peak_phase, peak_bin;
            float peak_mag;
            cuacq_find_peak(&d_cuda_state, effective_fft_size,
                &peak_phase, &peak_bin, &peak_mag);

            indext = peak_phase;
            d_mag = peak_mag;

            if (d_use_CFAR_algorithm_flag)
                {
                    float mean_power = cuacq_compute_input_power(&d_cuda_state, effective_fft_size);
                    d_test_statistics = (mean_power > 0.0f) ? (peak_mag / mean_power) : 0.0f;
                    d_input_power = mean_power;
                }
            else
                {
                    float second_peak = cuacq_second_peak(&d_cuda_state,
                        peak_bin, peak_phase, effective_fft_size, d_samplesPerChip);
                    d_test_statistics = (second_peak > 0.0f) ? (peak_mag / second_peak) : 0.0f;
                }

            doppler = static_cast<int32_t>(d_doppler_center_step_two -
                          (static_cast<float>(d_num_doppler_bins_step2) / 2.0f) * d_acq_parameters.doppler_step2 +
                          d_acq_parameters.doppler_step2 * static_cast<float>(peak_bin));

            if (d_acq_parameters.use_automatic_resampler)
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(std::fmod(static_cast<float>(indext), d_acq_parameters.samples_per_code)) * d_acq_parameters.resampler_ratio;
                    d_gnss_synchro->Acq_delay_samples -= static_cast<double>(d_acq_parameters.resampler_latency_samples);
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = rint(static_cast<double>(samp_count) * d_acq_parameters.resampler_ratio);
                    d_gnss_synchro->Acq_doppler_step = d_acq_parameters.doppler_step2;
                    d_gnss_synchro->fs = d_acq_parameters.resampled_fs;
                }
            else
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(std::fmod(static_cast<float>(indext), d_acq_parameters.samples_per_code));
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = samp_count;
                    d_gnss_synchro->Acq_doppler_step = d_acq_parameters.doppler_step2;
                    d_gnss_synchro->fs = d_acq_parameters.fs_in;
                }

            // Restore original Doppler params after step two
            cuacq_update_doppler_params(&d_cuda_state,
                d_num_doppler_bins,
                static_cast<float>(d_acq_parameters.doppler_max),
                static_cast<float>(d_doppler_step),
                static_cast<float>(d_doppler_center + d_doppler_bias));
        }

    if (d_acq_parameters.blocking)
        {
            lk.lock();
        }

    // Threshold decision and state transitions (same logic as pcps_acquisition)
    if (!d_acq_parameters.bit_transition_flag)
        {
            if (d_test_statistics > d_threshold)
                {
                    d_active = false;
                    if (d_acq_parameters.make_2_steps)
                        {
                            if (d_step_two)
                                {
                                    send_positive_acquisition();
                                    d_step_two = false;
                                    d_state = 0;
                                }
                            else
                                {
                                    d_step_two = true;
                                    d_doppler_center_step_two = static_cast<float>(d_gnss_synchro->Acq_doppler_hz);
                                    d_num_noncoherent_integrations_counter = 0;
                                    d_positive_acq = 0;
                                    d_state = 0;
                                }
                            calculate_threshold();
                        }
                    else
                        {
                            send_positive_acquisition();
                            d_state = 0;
                        }
                }
            else
                {
                    d_buffer_count = 0;
                    d_state = 1;
                }

            if (d_num_noncoherent_integrations_counter == d_acq_parameters.max_dwells)
                {
                    if (d_state != 0)
                        {
                            send_negative_acquisition();
                        }
                    d_state = 0;
                    d_active = false;
                    const bool was_step_two = d_step_two;
                    d_step_two = false;
                    if (was_step_two)
                        {
                            calculate_threshold();
                        }
                }
        }
    else
        {
            d_active = false;
            if (d_test_statistics > d_threshold)
                {
                    if (d_acq_parameters.make_2_steps)
                        {
                            if (d_step_two)
                                {
                                    send_positive_acquisition();
                                    d_step_two = false;
                                    d_state = 0;
                                }
                            else
                                {
                                    d_step_two = true;
                                    d_doppler_center_step_two = static_cast<float>(d_gnss_synchro->Acq_doppler_hz);
                                    d_num_noncoherent_integrations_counter = 0U;
                                    d_state = 0;
                                }
                            calculate_threshold();
                        }
                    else
                        {
                            send_positive_acquisition();
                            d_state = 0;
                        }
                }
            else
                {
                    d_state = 0;
                    const bool was_step_two = d_step_two;
                    d_step_two = false;
                    if (was_step_two)
                        {
                            calculate_threshold();
                        }
                    send_negative_acquisition();
                }
        }

    d_worker_active = false;

    if ((d_num_noncoherent_integrations_counter == d_acq_parameters.max_dwells) or (d_positive_acq == 1) or (d_acq_parameters.bit_transition_flag))
        {
            d_num_noncoherent_integrations_counter = 0U;
            d_positive_acq = 0;
        }
}


bool pcps_acquisition_cuda::start()
{
    gr::thread::scoped_lock lk(d_setlock);
    d_sample_counter = 0ULL;
    calculate_threshold();
    return true;
}


int pcps_acquisition_cuda::general_work(int noutput_items __attribute__((unused)),
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    gr::thread::scoped_lock lk(d_setlock);
    if (!d_active or d_worker_active)
        {
            bool consume_samples = ((!d_active) || (d_worker_active && (d_num_noncoherent_integrations_counter == d_acq_parameters.max_dwells)));
            if ((!d_acq_parameters.blocking_on_standby) && consume_samples)
                {
                    d_sample_counter += static_cast<uint64_t>(ninput_items[0]);
                    consume_each(ninput_items[0]);
                }
            if (d_step_two)
                {
                    d_state = 0;
                    d_active = true;
                }
            return 0;
        }

    switch (d_state)
        {
        case 0:
            {
                d_gnss_synchro->Acq_delay_samples = 0.0;
                d_gnss_synchro->Acq_doppler_hz = 0.0;
                d_gnss_synchro->Acq_samplestamp_samples = 0ULL;
                d_gnss_synchro->Acq_doppler_step = 0U;
                d_mag = 0.0;
                d_state = 1;
                d_buffer_count = 0U;
                if (!d_acq_parameters.blocking_on_standby)
                    {
                        d_sample_counter += static_cast<uint64_t>(ninput_items[0]);
                        consume_each(ninput_items[0]);
                    }
                break;
            }
        case 1:
            {
                uint32_t buff_increment;
                if (d_cshort)
                    {
                        const auto* in = reinterpret_cast<const lv_16sc_t*>(input_items[0]);
                        if ((ninput_items[0] + d_buffer_count) <= d_consumed_samples)
                            {
                                buff_increment = ninput_items[0];
                            }
                        else
                            {
                                buff_increment = d_consumed_samples - d_buffer_count;
                            }
                        std::copy(in, in + buff_increment, d_data_buffer_sc.begin() + d_buffer_count);
                    }
                else
                    {
                        const auto* in = reinterpret_cast<const gr_complex*>(input_items[0]);
                        if ((ninput_items[0] + d_buffer_count) <= d_consumed_samples)
                            {
                                buff_increment = ninput_items[0];
                            }
                        else
                            {
                                buff_increment = d_consumed_samples - d_buffer_count;
                            }
                        std::copy(in, in + buff_increment, d_data_buffer.begin() + d_buffer_count);
                    }

                if (d_buffer_count >= d_consumed_samples)
                    {
                        d_state = 2;
                    }
                d_buffer_count += buff_increment;
                d_sample_counter += static_cast<uint64_t>(buff_increment);
                consume_each(buff_increment);
                break;
            }
        case 2:
            {
                if (d_acq_parameters.blocking)
                    {
                        lk.unlock();
                        acquisition_core(d_sample_counter);
                    }
                else
                    {
                        gr::thread::thread d_worker(&pcps_acquisition_cuda::acquisition_core, this, d_sample_counter);
                        d_worker_active = true;
                    }
                consume_each(0);
                d_buffer_count = 0U;
                break;
            }
        }

    // Send outputs to the monitor
    if (d_acq_parameters.enable_monitor_output)
        {
            auto** out = reinterpret_cast<Gnss_Synchro**>(&output_items[0]);
            if (!d_monitor_queue.empty())
                {
                    int num_gnss_synchro_objects = d_monitor_queue.size();
                    for (int i = 0; i < num_gnss_synchro_objects; ++i)
                        {
                            Gnss_Synchro current_synchro_data = d_monitor_queue.front();
                            d_monitor_queue.pop();
                            *out[i] = std::move(current_synchro_data);
                        }
                    return num_gnss_synchro_objects;
                }
        }

    return 0;
}
