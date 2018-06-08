#ifndef CONTRIBUTION_INTERFACE_H
#define CONTRIBUTION_INTERFACE_H

#include <cstddef>

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// @brief Interface to contribution of any type around an interaction loop
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class IContribution
{
public:
  /// @brief Destructor of IContribution
  virtual ~IContribution() = default;

  /// @brief Preprocessing contribution (before the loop over each particle)
  virtual void preprocess() = 0;

  /// @brief Main contribution (inside the loop over each particle)
  /// @param[in] i: the index of the processed particle
  virtual void compute(size_t const i) = 0;

  /// @brief Postprocessing contribution (after the loop over each particle)
  virtual void postprocess() = 0;
};

#endif // CONTRIBUTION_INTERFACE_H
