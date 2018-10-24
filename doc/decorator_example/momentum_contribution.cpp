#include "momentum_contribution.h"

#include <iostream>

void MomentumContribution::preprocess()
{
  // TODO
}

void MomentumContribution::compute(size_t const i)
{
	m_contributionsPtr->compute(i);
  	std::cout << "Momentum(" << i << ") " << *m_exampleNumberPtr << std::endl;
}

void MomentumContribution::postprocess()
{
  // TODO
}