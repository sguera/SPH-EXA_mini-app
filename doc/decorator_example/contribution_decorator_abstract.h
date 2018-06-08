#ifndef CONTRIBUTION_DECORATOR_ABSTRACT_H
#define CONTRIBUTION_DECORATOR_ABSTRACT_H

#include <memory>

#include "contribution_interface.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// @brief Abstract class to decorate computation steps
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class AContributionDecorator : public IContribution
{
public:

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Functions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// @brief Destructor of AContributionDecorator
  virtual ~AContributionDecorator() = default;

  /// @brief Preprocessing contribution (before the loop over each particle)
  virtual void preprocess() override = 0;

  /// @brief Main contribution (inside the loop over each particle)
  /// @param[in] i: the index of the processed particle
  virtual void compute(size_t const i) override = 0;

  /// @brief Postprocessing contribution (after the loop over each particle)
  virtual void postprocess() override = 0;

protected:

  /// @brief Constructor of AContributionDecorator
  /// @param[in] contributionsPtr: a pointer to the already attached contributions
  AContributionDecorator(std::unique_ptr<IContribution> contributionsPtr)
    : m_contributionsPtr{std::move(contributionsPtr)} {}

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Attributes
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// @brief Pointer to already attached contributions
  std::shared_ptr<IContribution> m_contributionsPtr;
};

#endif // CONTRIBUTION_DECORATOR_ABSTRACT_H
