#ifndef TRTPLUGIN_H
#define TRTPLUGIN_H

#include "trtplugin_global.h"

#ifdef __cplusplus
extern "C" {
#endif

TRTPLUGIN_API bool initLibNvInferPlugins(void* logger, const char* libNamespace);

#ifdef __cplusplus
};
#endif

#endif // TRTPLUGIN_H
