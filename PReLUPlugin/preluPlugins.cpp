/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "preluPlugins.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>

using namespace nvinfer1;
using nvinfer1::plugin::PReLU;
using nvinfer1::plugin::PReLUPluginCreator;

namespace
{
const char* PRELU_PLUGIN_VERSION{"1"};
const char* PRELU_PLUGIN_NAME{"PReLU"};
} // namespace

//FIX
PluginFieldCollection PReLUPluginCreator::mFC{};
std::vector<PluginField> PReLUPluginCreator::mPluginAttributes;


PReLU::PReLU(const Weights *weights, int nbWeights)
{
    // since we want to deal with the case where there is no bias, we can't infer
    // the number of channels from the bias weights.
    assert(nbWeights == 1);
    mNbWeights = nbWeights;
    mPReLuWeights = copyToDevice(weights[0].values, weights[0].count);
}

PReLU::PReLU(const void *buffer, size_t length)
{
    const char* d = reinterpret_cast<const char*>(buffer), *a = d;
    width = read<int>(d);
    height = read<int>(d);
    channel = read<int>(d);
    mPReLuWeights = deserializeToDevice(d, channel);
    assert(d == a + length);
}

PReLU::~PReLU()
{
    cudaFree(const_cast<void*>(mPReLuWeights.values));
    cudaFree(deviceData);
}

int PReLU::getNbOutputs() const
{
    return 1;
}

Dims PReLU::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    width = inputs[0].d[2];
    height = inputs[0].d[1];
    channel = inputs[0].d[0];
    //PReLu output dims the same as input dims
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int PReLU::initialize()
{
    return 0;
}

void PReLU::terminate()
{

}

size_t PReLU::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int PReLU::enqueue(int batchSize, const void * const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    calcPReLU(reinterpret_cast<const float *>(inputs[0]), (float*)outputs[0],
            reinterpret_cast<const float*>(mPReLuWeights.values),
            batchSize, mPReLuWeights.count, width, height, stream);
    return 0;
}

size_t PReLU::getSerializationSize() const
{
    return sizeof(int) * 3 + mPReLuWeights.count * sizeof(float);
}

void PReLU::serialize(void *buffer) const
{
    char* d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, width);
    write(d, height);
    write(d, channel);
    serializeFromDevice(d, mPReLuWeights);
    assert(d == a + getSerializationSize());
}

bool PReLU::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == PluginFormat::kNCHW;
}

const char *PReLU::getPluginType() const
{
    return PRELU_PLUGIN_NAME;
}

const char *PReLU::getPluginVersion() const
{
    return PRELU_PLUGIN_VERSION;
}

void PReLU::destroy()
{
    delete this;
}

IPluginV2Ext *PReLU::clone() const
{
    // Create a new instance
    IPluginV2Ext* plugin = new PReLU(&mPReLuWeights, mNbWeights);

    // Set the namespace
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void PReLU::setPluginNamespace(const char *pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char *PReLU::getPluginNamespace() const
{
    return mPluginNamespace;
}

DataType PReLU::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    return DataType::kFLOAT;
}

bool PReLU::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool PReLU::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void PReLU::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator)
{

}

void PReLU::configurePlugin(const Dims *inputDims, int nbInputs,
                            const Dims *outputDims, int nbOutputs,
                            const DataType *inputTypes, const DataType *outputTypes,
                            const bool *inputIsBroadcast, const bool *outputIsBroadcast,
                            PluginFormat floatFormat, int maxBatchSize)
{
    assert((*inputTypes == nvinfer1::DataType::kFLOAT || *inputTypes == nvinfer1::DataType::kHALF)
           && floatFormat == PluginFormat::kNCHW);
    mDataType = *inputTypes;
}

void PReLU::detachFromContext()
{

}

template<typename T> T PReLU::read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

Weights PReLU::copyToDevice(const void* hostData, size_t count)
{
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{ nvinfer1::DataType::kFLOAT, deviceData, int64_t(count) };
}

void PReLU::serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float),
               cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights PReLU::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

PReLUPluginCreator::PReLUPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nbWeights", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *PReLUPluginCreator::getPluginName() const
{
    return PRELU_PLUGIN_NAME;
}

const char *PReLUPluginCreator::getPluginVersion() const
{
    return PRELU_PLUGIN_VERSION;
}

const PluginFieldCollection *PReLUPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext *PReLUPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc)
{
    std::vector<float> weightValues;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "nbWeights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mNbWeights = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            weightValues.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                weightValues.push_back(*w);
                w++;
            }
        }
    }
    Weights weights{DataType::kFLOAT, weightValues.data(), (int64_t) weightValues.size()};

    PReLU* obj = new PReLU(&weights, mNbWeights);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext *PReLUPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new PReLU(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
