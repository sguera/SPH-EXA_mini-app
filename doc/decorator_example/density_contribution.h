#ifndef DENSITY_CONTRIBUTION_H
#define DENSITY_CONTRIBUTION_H

#include <memory>

#include "contribution_decorator_abstract.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// @brief Density contribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DensityContribution : public AContributionDecorator
{
public:

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Functions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// @brief Constructor of DensityContribution
  /// @param[in] contributionsPtr: a pointer to the already attached contributions
  /// @param[in] exampleArg: a pointer to a number for the example
  DensityContribution(std::unique_ptr<IContribution> contributionsPtr, // DO NOT REMOVE
                      std::shared_ptr<int> exampleArg)
    : AContributionDecorator(std::move(contributionsPtr)), // DO NOT REMOVE
    m_exampleNumber(exampleArg) {} // copy the pointer to the number in DensityContribution attribtues so that it becomes accessible to DensityContribution during the whole simulation

  /// @brief Destructor of DensityContribution
  virtual ~DensityContribution() = default;

  /// @brief Preprocessing contribution (before the loop over each particle)
  virtual void preprocess() override;

  /// @brief Main contribution (inside the loop over each particle)
  /// @param[in] i: the index of the processed particle
  virtual void compute(size_t const i) override;

  /// @brief Postprocessing contribution (after the loop over each particle)
  virtual void postprocess() override;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Attributes
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// TODO Required attribtues to compute the density contribution (i.e. SPHOperator, EquationOfState, etc.)
  std::shared_ptr<int> m_exampleNumberPtr;
};


#endif // DENSITY_CONTRIBUTION_H
