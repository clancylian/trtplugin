#ifndef TRTPLUGIN_GLOBAL_H
#define TRTPLUGIN_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(TRTPLUGIN_LIBRARY)
#  define TRTPLUGIN_API Q_DECL_EXPORT
#else
#  define TRTPLUGIN_API Q_DECL_IMPORT
#endif

#endif // TRTPLUGIN_GLOBAL_H
