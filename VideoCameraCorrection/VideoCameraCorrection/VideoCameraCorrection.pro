TEMPLATE = app
CONFIG += console c++11
CONFIG += -O0
CONFIG += -std=c++17
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += /usr/local/include/opencv4
LIBS += -L/usr/local/lib \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_calib3d \
        -lopencv_highgui \
        -lopencv_videostab \
        -lopencv_imgproc \
        -lopencv_features2d \
        -lopencv_optflow \
        -lopencv_tracking

LIBS += -lstdc++fs

SOURCES += \
        cmc.cpp \
        main.cpp

HEADERS += \
    cmc.h

