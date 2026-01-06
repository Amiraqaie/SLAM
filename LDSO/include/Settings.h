#ifndef LDSO_SETTING_H
#define LDSO_SETTING_H

// a lot of parameters can be set in LDSO
namespace ldso{

    const int PYR_LEVELS = 6;
    const int NUM_THREADS = 6;

    // the config bits in solver
    const int SOLVER_SVD = 1;
    const int SOLVER_ORTHOGONALIZE_SYSTEM = 2;
    const int SOLVER_ORTHOGONALIZE_POINTMARG = 4;
    const int SOLVER_ORTHOGONALIZE_FULL = 8;
    const int SOLVER_SVD_CUT7 = 16;
    const int SOLVER_REMOVE_POSTPRIOR = 32;
    const int SOLVER_USE_GN = 64;
    const int SOLVER_FIX_LAMBDA = 128;
    const int SOLVER_ORTHOGONALIZE_X = 256;
    const int SOLVER_MOMENTUM = 512;
    const int SOLVER_STEPMOMENTUM = 1024;
    const int SOLVER_ORTHOGONALIZE_X_LATER = 2048;

    // constants to scale that parameters in optimization
    const float SCALE_IDEPTH = 1.0f;
    const float SCALE_XI_ROT = 1.0f;
    const float SCALE_XI_TRANS = 0.5f;
    const float SCALE_F = 50.0f;
    const float SCALE_C = 50.0f;
    const float SCALE_W = 1.0f;
    const float SCALE_A = 10.0f;
    const float SCALE_B = 1000.0f;

    // the detail setting variables
    extern int pyrLevelsUsed;
    extern int setting_keyframesPerSecond;
    extern int setting_realTimeMaxKF;
    extern int setting_maxShiftWeightT;
    extern int setting_maxShiftWeightR;
    extern int setting_maxShiftWeightRT;

    extern float setting_maxAffineWeight;
    extern float setting_kfGlobalweight;
    extern float setting_idepthFixPrior;
    extern float setting_idepthFixPriorMargFac;
    extern float setting_initialRotPrior;
    extern float setting_initialAffBPrior;
    extern float setting_initialAffBPrior;


    extern float setting_desiredImmatureDensity;

}

#endif