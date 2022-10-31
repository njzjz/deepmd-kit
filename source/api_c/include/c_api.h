#ifdef __cplusplus
extern "C" {
#endif

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

/**
* @brief The deep potential.
**/
typedef struct DP_DeepPot DP_DeepPot;

/**
* @brief DP constructor with initialization.
* @param[in] c_model The name of the frozen model file.
**/
extern DP_DeepPot* DP_NewDeepPot(const char* c_model);

/**
* @brief Evaluate the energy, force and virial by using a DP.
* @param[in] dp The DP to use.
* @param[in] natoms The number of atoms.
* @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
* @param[in] atype The atom types. The array should contain natoms ints.
* @param[in] box The cell of the region. The array should be of size nframes x 9.
* @param[out] energy Output energy.
* @param[out] force Output force.
* @param[out] virial Output virial.
  **/
extern void DP_DeepPotCompute (
  DP_DeepPot* dp,
  const int natom,
  const VALUETYPE* coord,
  const int* atype,
  const VALUETYPE* cell,
  const ENERGYTYPE* energy,
  const VALUETYPE* force,
  const VALUETYPE* virial
  );

#ifdef __cplusplus
} /* end extern "C" */
#endif