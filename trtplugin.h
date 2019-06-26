#ifndef TRTPLUGIN_H
#define TRTPLUGIN_H

#include "trtplugin_global.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifdef __cplusplus
extern "C" {
#endif

TRTPLUGIN_API bool initLibNvInferPluginsExt(void* logger, const char* libNamespace);

#ifdef __cplusplus
};
#endif

#endif // TRTPLUGIN_H
