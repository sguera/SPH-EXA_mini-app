#ifndef CORE_CONTRIBUTION_H
#define CORE_CONTRIBUTION_H

#include "contribution_interface.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// @brief Core of a contribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class CoreComputationStep : public IComputationStep
{
public:

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Functions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// @brief Constructor of CoreComputationStep
  CoreComputationStep() = default;

    /// @brief Destructor of CoreContribution
  virtual ~CoreContribution() = default;

  /// @brief Preprocessing contribution (before the loop over each particle)
  virtual void preprocess() override {}

  /// @brief Main contribution (inside the loop over each particle)
  /// @param[in] i: the index of the processed particle
  virtual void compute(size_t const i) override {}

  /// @brief Postprocessing contribution (after the loop over each particle)
  virtual void postprocess() override {}
};

#endif // CORE_CONTRIBUTION_H
