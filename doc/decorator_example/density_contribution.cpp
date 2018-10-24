#include "density_contribution.h"

#include <iostream>

void DensityContribution::preprocess()
{
  // TODO
}

void DensityContribution::compute(size_t const i)
{
	m_contributionsPtr->compute(i);
  	std::cout << "Density(" << i << ") " << *m_exampleNumberPtr << std::endl;
}

void DensityContribution::postprocess()
{
  // TODO
}
