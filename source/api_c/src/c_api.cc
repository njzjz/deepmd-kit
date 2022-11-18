#include "c_api.h"

#include <vector>
#include <string>
#include <numeric>
#include "c_api_internal.h"
#include "common.h"
#include "DeepPot.h"

extern "C" {

DP_DeepPot::DP_DeepPot(deepmd::DeepPot& dp)
    : dp(dp) {}

DP_DeepPot* DP_NewDeepPot(const char* c_model) {
    std::string model(c_model);
    deepmd::DeepPot dp(model);
    DP_DeepPot* new_dp = new DP_DeepPot(dp);
    return new_dp;
}

DP_DeepPotModelDevi::DP_DeepPotModelDevi(deepmd::DeepPotModelDevi& dp)
    : dp(dp) {}

DP_DeepPotModelDevi* DP_NewDeepPotModelDevi(const char** c_models, int n_models) {
    std::vector<std::string> model(c_models, c_models + n_models);
    deepmd::DeepPotModelDevi dp(model);
    DP_DeepPotModelDevi* new_dp = new DP_DeepPotModelDevi(dp);
    return new_dp;
}
} // extern "C"

template <typename VALUETYPE>
void DP_DeepPotCompute_variant (
    DP_DeepPot* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    double e;
    std::vector<VALUETYPE> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_);
    // copy from C++ vectors to C arrays, if not NULL pointer
    if(energy) *energy = e;
    if(force) std::copy(f.begin(), f.end(), force);
    if(virial) std::copy(v.begin(), v.end(), virial);
    if(atomic_energy) std::copy(ae.begin(), ae.end(), atomic_energy);
    if(atomic_virial) std::copy(av.begin(), av.end(), atomic_virial);
}

template
void DP_DeepPotCompute_variant <double> (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    );

template
void DP_DeepPotCompute_variant <float> (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    );


// https://stackoverflow.com/a/64396974/9567349
template <typename VALUETYPE>
inline vector<VALUETYPE> flatten_vector(const std::vector<std::vector<VALUETYPE>>& v) {
    return accumulate(v.begin(), v.end(),
        vector<VALUETYPE>(),
        [](vector<VALUETYPE>& a, vector<VALUETYPE>& b) {
            a.insert(a.end(), b.begin(), b.end());
            return a;
        });
}


template <typename VALUETYPE>
void DP_DeepPotModelDeviCompute_variant (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    double* energy,
    VALUETYPE* force,
    VALUETYPE* virial,
    VALUETYPE* atomic_energy,
    VALUETYPE* atomic_virial
    ) {
    // init C++ vectors from C arrays
    std::vector<VALUETYPE> coord_(coord, coord+natoms*3);
    std::vector<int> atype_(atype, atype+natoms);
    std::vector<VALUETYPE> cell_;
    if (cell) {
        // pbc
        cell_.assign(cell, cell+9);
    }
    // different from DeepPot
    std::vector<double> e;
    std::vector<std::vector<VALUETYPE>> f, v, ae, av;

    dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_);
    // 2D vector to 2D array, flatten first    
    if(energy) {
        e_flat = flatten_vector(e);
        std::copy(e_flat.begin(), e_flat.end(), energy);
    }
    if(force) {
        f_flat = flatten_vector(f);
        std::copy(f_flat.begin(), f_flat.end(), force);
    }
    if(virial) {
        v_flat = flatten_vector(v);
        std::copy(v_flat.begin(), v_flat.end(), virial);
    }
    if(atomic_energy) {
        ae_flat = flatten_vector(ae);
        std::copy(ae_flat.begin(), ae_flat.end(), atomic_energy);
    }
    if(atomic_virial) {
        av_flat = flatten_vector(av);
        std::copy(av_flat.begin(), av_flat.end(), atomic_virial);
    }
}

template
void DP_DeepPotModelDeviCompute_variant <double> (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    );

template
void DP_DeepPotModelDeviCompute_variant <float> (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    );

extern "C" {

void DP_DeepPotCompute (
    DP_DeepPot* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    ) {
    DP_DeepPotCompute_variant<double>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputef (
    DP_DeepPot* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    ) {
    DP_DeepPotCompute_variant<float>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

const char* DP_DeepPotGetTypeMap(
    DP_DeepPot* dp
    ) {
    std::string type_map;
    dp->dp.get_type_map(type_map);
    // copy from string to char*
    const std::string::size_type size = type_map.size();
    // +1 for '\0'
    char *buffer = new char[size + 1];
    std::copy(type_map.begin(), type_map.end(), buffer);
    buffer[size] = '\0';
    return buffer;
}

double DP_DeepPotGetCutoff(
    DP_DeepPot* dp
    ) {
    return dp->dp.cutoff();
}

int DP_DeepPotGetNumbTypes(
    DP_DeepPot* dp
    ) {
    return dp->dp.numb_types();
}

void DP_DeepPotModelDeviCompute (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial
    ) {
    DP_DeepPotModelDeviCompute_variant<double>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputef (
    DP_DeepPotModelDevi* dp,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial
    ) {
    DP_DeepPotModelDeviCompute_variant<float>(dp, natoms, coord, atype, cell, energy, force, virial, atomic_energy, atomic_virial);
}

double DP_DeepPotModelDeviGetCutoff(
    DP_DeepPotModelDevi* dp
    ) {
    return dp->dp.cutoff();
}

int DP_DeepPotModelDeviGetNumbTypes(
    DP_DeepPotModelDevi* dp
    ) {
    return dp->dp.numb_types();
}

void DP_ConvertPbtxtToPb(
    const char* c_pbtxt,
    const char* c_pb
    ) {
    std::string pbtxt(c_pbtxt);
    std::string pb(c_pb);
    deepmd::convert_pbtxt_to_pb(pbtxt, pb);
}

} // extern "C"