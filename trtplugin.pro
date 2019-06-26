#-------------------------------------------------
#
# Project created by QtCreator 2019-06-20T15:07:54
#
#-------------------------------------------------

QT       -= gui
CONFIG += c++11 console

TARGET = trtplugin
TEMPLATE = lib

DEFINES += TRTPLUGIN_LIBRARY

SOURCES += trtplugin.cpp \
    reorgPlugin/reorgPlugin.cpp \
    regionPlugin/regionPlugin.cpp \
    proposalPlugin/proposalPlugin.cpp \
    priorBoxPlugin/priorBoxPlugin.cpp \
    PReLUPlugin/preluPlugins.cpp \
    batchedNMSPlugin/batchedNMSInference.cpp \
    batchedNMSPlugin/batchedNMSPlugin.cpp \
    common/kernels/kernel.cpp \
    common/nmsHelper.cpp \
    cropAndResizePlugin/cropAndResizePlugin.cpp \
    flattenConcat/flattenConcat.cpp \
    gridAnchorPlugin/gridAnchorPlugin.cpp \
    nmsPlugin/nmsPlugin.cpp \
    normalizePlugin/normalizePlugin.cpp \
    nvFasterRCNN/nvFasterRCNNPlugin.cpp

HEADERS += trtplugin.h\
        trtplugin_global.h \
    reorgPlugin/reorgPlugin.h \
    regionPlugin/regionPlugin.h \
    proposalPlugin/proposalPlugin.h \
    priorBoxPlugin/priorBoxPlugin.h \
    PReLUPlugin/preluPlugins.h \
    batchedNMSPlugin/batchedNMSInference.h \
    batchedNMSPlugin/batchedNMSPlugin.h \
    batchedNMSPlugin/gatherNMSOutputs.h \
    common/kernels/kernel.h \
    common/kernels/reducedMath.h \
    common/bboxUtils.h \
    common/cub_helper.h \
    common/nmsUtils.h \
    common/plugin.h \
    cropAndResizePlugin/cropAndResizePlugin.h \
    flattenConcat/flattenConcat.h \
    gridAnchorPlugin/gridAnchorPlugin.h \
    nmsPlugin/nmsPlugin.h \
    normalizePlugin/normalizePlugin.h \
    nvFasterRCNN/nvFasterRCNNPlugin.h

CUDA_SOURCES += \
    batchedNMSPlugin/gatherNMSOutputs.cu \
    common/kernels/allClassNMS.cu \
    common/kernels/bboxDeltas2Proposals.cu \
    common/kernels/calcprelu.cu \
    common/kernels/common.cu \
    common/kernels/cropAndResizeKernel.cu \
    common/kernels/decodeBBoxes.cu \
    common/kernels/detectionForward.cu \
    common/kernels/extractFgScores.cu \
    common/kernels/gatherTopDetections.cu \
    common/kernels/generateAnchors.cu \
    common/kernels/gridAnchorLayer.cu \
    common/kernels/nmsLayer.cu \
    common/kernels/normalizeLayer.cu \
    common/kernels/permuteData.cu \
    common/kernels/priorBoxLayer.cu \
    common/kernels/proposalKernel.cu \
    common/kernels/proposalsForward.cu \
    common/kernels/regionForward.cu \
    common/kernels/reorgForward.cu \
    common/kernels/roiPooling.cu \
    common/kernels/rproiInferenceFused.cu \
    common/kernels/sortScoresPerClass.cu \
    common/kernels/sortScoresPerImage.cu

unix {
    target.path = /usr/lib
    INSTALLS += target
}

INCLUDEPATH += \
    /home/ubuntu/trtplugins/common \
    /home/ubuntu/trtplugins/common/kernels \
    /home/ubuntu/trtplugins/cub

INCLUDEPATH += /usr/local/TensorRT/include
LIBS += -L/usr/local/TensorRT/lib -lnvinfer -lnvcaffe_parser -lnvinfer_plugin

unix {
    CUDA_DIR = /usr/local/cuda-10.1
    SYSTEM_TYPE = 64            #操作系统位数 '32' or '64',
    CUDA_ARCH = sm_61         # cuda架构, for example 'compute_10', 'compute_11', 'sm_10'
    NVCC_OPTIONS = -lineinfo -Xcompiler -fPIC -std=c++11

    INCLUDEPATH += $$CUDA_DIR/include
    LIBS += -L$$CUDA_DIR/lib64
    LIBS += -lcuda -lcudart -lcudnn -lcurand
    # npp
    LIBS += -lnppig -lnppicc -lnppc -lnppidei -lnppist

    CUDA_OBJECTS_DIR = ./

    # The following makes sure all path names (which often include spaces) are put between quotation marks
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

    CONFIG(debug, debug|release): {
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
    } else:CONFIG(release, debug|release): {
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
        cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE \
            -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }
}
