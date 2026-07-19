#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <c10/util/Half.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "selective_scan_oflex.h"

template <int knrows, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase& params, cudaStream_t stream);

namespace
{
constexpr char const* kPLUGIN_NAME{"SelectiveScan"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kINPUTS{7};

size_t stateWorkspaceBytes(nvinfer1::Dims const& u, nvinfer1::Dims const& a)
{
    if (u.nbDims != 3 || a.nbDims != 2 || u.d[0] < 1 || u.d[1] < 1 || u.d[2] < 1 || a.d[1] < 1)
    {
        return 0;
    }
    int64_t const chunks = (static_cast<int64_t>(u.d[2]) + 2047) / 2048;
    return static_cast<size_t>(u.d[0]) * u.d[1] * chunks * a.d[1] * 2 * sizeof(float);
}

class SelectiveScanPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    SelectiveScanPlugin(bool deltaSoftplus, bool outFloat)
        : mDeltaSoftplus(deltaSoftplus)
        , mOutFloat(outFloat)
    {
    }

    SelectiveScanPlugin(void const* data, size_t length)
    {
        if (length == 2 * sizeof(int32_t))
        {
            auto const* values = static_cast<int32_t const*>(data);
            mDeltaSoftplus = values[0] != 0;
            mOutFloat = values[1] != 0;
        }
    }

    char const* getPluginType() const noexcept override { return kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

    size_t getSerializationSize() const noexcept override { return 2 * sizeof(int32_t); }

    void serialize(void* buffer) const noexcept override
    {
        int32_t values[2] = {static_cast<int32_t>(mDeltaSoftplus), static_cast<int32_t>(mOutFloat)};
        std::memcpy(buffer, values, sizeof(values));
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* plugin = new SelectiveScanPlugin(mDeltaSoftplus, mOutFloat);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::IExprBuilder&) noexcept override
    {
        if (outputIndex != 0 || nbInputs != kINPUTS)
        {
            return {};
        }
        return inputs[0];
    }

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (index != 0 || nbInputs != kINPUTS)
        {
            return nvinfer1::DataType::kFLOAT;
        }
        return mOutFloat ? nvinfer1::DataType::kFLOAT : inputTypes[0];
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (nbInputs != kINPUTS || nbOutputs != 1 || pos < 0 || pos >= nbInputs + nbOutputs)
        {
            return false;
        }
        if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        {
            return false;
        }
        auto const dynamicType = inOut[0].type;
        if (pos == 0)
        {
            return dynamicType == nvinfer1::DataType::kFLOAT || dynamicType == nvinfer1::DataType::kHALF;
        }
        if (pos == 1 || pos == 3 || pos == 4)
        {
            return inOut[pos].type == dynamicType;
        }
        if (pos == 2 || pos == 5 || pos == 6)
        {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }
        return inOut[pos].type == (mOutFloat ? nvinfer1::DataType::kFLOAT : dynamicType);
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*,
        int32_t) noexcept override
    {
    }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override
    {
        return nbInputs == kINPUTS ? stateWorkspaceBytes(inputs[0].dims, inputs[2].dims) : 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        auto const& u = inputDesc[0].dims;
        auto const& delta = inputDesc[1].dims;
        auto const& a = inputDesc[2].dims;
        auto const& b = inputDesc[3].dims;
        auto const& c = inputDesc[4].dims;
        if (u.nbDims != 3 || delta.nbDims != 3 || a.nbDims != 2 || b.nbDims != 4 || c.nbDims != 4
            || u.d[0] != delta.d[0] || u.d[2] != delta.d[2] || u.d[1] != a.d[0]
            || b.d[0] != u.d[0] || c.d[0] != u.d[0] || b.d[1] != c.d[1] || b.d[2] != a.d[1]
            || c.d[2] != a.d[1] || b.d[3] != u.d[2] || c.d[3] != u.d[2] || b.d[1] < 1
            || u.d[1] % b.d[1] != 0 || delta.d[1] < 1 || u.d[1] % delta.d[1] != 0 || workspace == nullptr)
        {
            return 1;
        }

        SSMParamsBase params{};
        params.batch = u.d[0];
        params.dim = u.d[1];
        params.seqlen = u.d[2];
        params.dstate = a.d[1];
        params.n_groups = b.d[1];
        params.n_chunks = (params.seqlen + 2047) / 2048;
        params.dim_ngroups_ratio = params.dim / params.n_groups;
        params.dim_deltagroups_ratio = params.dim / delta.d[1];
        params.delta_softplus = mDeltaSoftplus;

        params.A_d_stride = a.d[1];
        params.A_dstate_stride = 1;
        params.B_batch_stride = b.d[1] * b.d[2] * b.d[3];
        params.B_group_stride = b.d[2] * b.d[3];
        params.B_dstate_stride = b.d[3];
        params.C_batch_stride = c.d[1] * c.d[2] * c.d[3];
        params.C_group_stride = c.d[2] * c.d[3];
        params.C_dstate_stride = c.d[3];
        params.u_batch_stride = u.d[1] * u.d[2];
        params.u_d_stride = u.d[2];
        params.delta_batch_stride = delta.d[1] * delta.d[2];
        params.delta_d_stride = delta.d[2];
        params.out_batch_stride = params.u_batch_stride;
        params.out_d_stride = params.u_d_stride;

        params.u_ptr = const_cast<void*>(inputs[0]);
        params.delta_ptr = const_cast<void*>(inputs[1]);
        params.A_ptr = const_cast<void*>(inputs[2]);
        params.B_ptr = const_cast<void*>(inputs[3]);
        params.C_ptr = const_cast<void*>(inputs[4]);
        params.D_ptr = const_cast<void*>(inputs[5]);
        params.delta_bias_ptr = const_cast<void*>(inputs[6]);
        params.out_ptr = outputs[0];
        params.x_ptr = workspace;

        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            selective_scan_fwd_cuda<1, float, float, float>(params, stream);
        }
        else if (mOutFloat)
        {
            selective_scan_fwd_cuda<1, at::Half, float, float>(params, stream);
        }
        else
        {
            selective_scan_fwd_cuda<1, at::Half, float, at::Half>(params, stream);
        }
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

private:
    bool mDeltaSoftplus{true};
    bool mOutFloat{false};
    std::string mNamespace;
};

class SelectiveScanPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    SelectiveScanPluginCreator()
    {
        mFields.emplace_back("delta_softplus", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        mFields.emplace_back("out_float", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    char const* getPluginName() const noexcept override { return kPLUGIN_NAME; }
    char const* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }

    nvinfer1::IPluginV2* createPlugin(
        char const*, nvinfer1::PluginFieldCollection const* fields) noexcept override
    {
        bool deltaSoftplus = true;
        bool outFloat = false;
        if (fields != nullptr)
        {
            for (int32_t i = 0; i < fields->nbFields; ++i)
            {
                auto const& field = fields->fields[i];
                if (field.data == nullptr)
                {
                    continue;
                }
                if (std::strcmp(field.name, "delta_softplus") == 0)
                {
                    deltaSoftplus = *static_cast<int32_t const*>(field.data) != 0;
                }
                else if (std::strcmp(field.name, "out_float") == 0)
                {
                    outFloat = *static_cast<int32_t const*>(field.data) != 0;
                }
            }
        }
        auto* plugin = new SelectiveScanPlugin(deltaSoftplus, outFloat);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* serialData, size_t serialLength) noexcept override
    {
        auto* plugin = new SelectiveScanPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::vector<nvinfer1::PluginField> mFields;
    nvinfer1::PluginFieldCollection mCollection{};
    std::string mNamespace;
};
}

REGISTER_TENSORRT_PLUGIN(SelectiveScanPluginCreator);

extern "C" char const* depthart_selective_scan_trt_version()
{
    return "SelectiveScan-1";
}

