/*!
 * \file gps_l1_ca_pcps_acquisition_cuda.h
 * \brief Adapts a cuFFT PCPS acquisition block to AcquisitionInterface for
 *  GPS L1 C/A signals
 *
 * Config file usage:
 *   Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition_CUDA
 *
 * \author gnss-station project, 2026
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef GNSS_SDR_GPS_L1_CA_PCPS_ACQUISITION_CUDA_H
#define GNSS_SDR_GPS_L1_CA_PCPS_ACQUISITION_CUDA_H

#include "acq_conf.h"
#include "channel_fsm.h"
#include "complex_byte_to_float_x2.h"
#include "gnss_synchro.h"
#include "pcps_acquisition_cuda.h"
#include <gnuradio/blocks/float_to_complex.h>
#include <volk_gnsssdr/volk_gnsssdr_alloc.h>
#include <memory>
#include <string>
#include <utility>

/** \addtogroup Acquisition
 * \{ */
/** \addtogroup Acq_adapters acquisition_adapters
 * \{ */


class ConfigurationInterface;

/*!
 * \brief Adapts a cuFFT PCPS acquisition block to AcquisitionInterface for
 *  GPS L1 C/A signals
 */
class GpsL1CaPcpsAcquisitionCuda : public AcquisitionInterface
{
public:
    GpsL1CaPcpsAcquisitionCuda(
        const ConfigurationInterface* configuration,
        const std::string& role,
        unsigned int in_streams,
        unsigned int out_streams);

    ~GpsL1CaPcpsAcquisitionCuda() = default;

    inline std::string role() override
    {
        return role_;
    }

    inline std::string implementation() override
    {
        return "GPS_L1_CA_PCPS_Acquisition_CUDA";
    }

    inline size_t item_size() override
    {
        return item_size_;
    }

    void connect(gr::top_block_sptr top_block) override;
    void disconnect(gr::top_block_sptr top_block) override;
    gr::basic_block_sptr get_left_block() override;
    gr::basic_block_sptr get_right_block() override;

    void set_gnss_synchro(Gnss_Synchro* p_gnss_synchro) override;

    inline void set_channel(unsigned int channel) override
    {
        channel_ = channel;
        acquisition_->set_channel(channel_);
    }

    inline void set_channel_fsm(std::weak_ptr<ChannelFsm> channel_fsm) override
    {
        channel_fsm_ = std::move(channel_fsm);
        acquisition_->set_channel_fsm(channel_fsm_);
    }

    void set_threshold(float threshold) override;
    void set_doppler_max(unsigned int doppler_max) override;
    void set_doppler_step(unsigned int doppler_step) override;
    void set_doppler_center(int doppler_center) override;
    void init() override;
    void set_local_code() override;
    signed int mag() override;
    void reset() override;
    void set_state(int state) override;
    void stop_acquisition() override;
    void set_resampler_latency(uint32_t latency_samples) override;

private:
    pcps_acquisition_cuda_sptr acquisition_;
    volk_gnsssdr::vector<std::complex<float>> code_;
    std::weak_ptr<ChannelFsm> channel_fsm_;
    gr::blocks::float_to_complex::sptr float_to_complex_;
    complex_byte_to_float_x2_sptr cbyte_to_float_x2_;
    Gnss_Synchro* gnss_synchro_;
    Acq_Conf acq_parameters_;
    std::string item_type_;
    std::string dump_filename_;
    std::string role_;
    size_t item_size_;
    float threshold_;
    int doppler_center_;
    unsigned int vector_length_;
    unsigned int code_length_;
    unsigned int channel_;
    unsigned int doppler_max_;
    unsigned int doppler_step_;
    unsigned int sampled_ms_;
    unsigned int in_streams_;
    unsigned int out_streams_;
};


/** \} */
/** \} */
#endif  // GNSS_SDR_GPS_L1_CA_PCPS_ACQUISITION_CUDA_H
