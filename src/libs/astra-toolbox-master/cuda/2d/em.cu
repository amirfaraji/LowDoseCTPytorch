/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#include "astra/cuda/2d/em.h"
#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"

#include <cstdio>
#include <cassert>

namespace astraCUDA {


// TODO: ensure non-negativity somewhere??


EM::EM()
{
	D_projData = 0;
	D_tmpData = 0;
	D_pixelWeight = 0;

}


EM::~EM()
{
	reset();
}

void EM::reset()
{
	cudaFree(D_projData);
	cudaFree(D_tmpData);
	cudaFree(D_pixelWeight);

	D_projData = 0;
	D_tmpData = 0;
	D_pixelWeight = 0;

	ReconAlgo::reset();
}


bool EM::init()
{
	allocateVolumeData(D_pixelWeight, pixelPitch, dims);
	zeroVolumeData(D_pixelWeight, pixelPitch, dims);

	allocateVolumeData(D_tmpData, tmpPitch, dims);
	zeroVolumeData(D_tmpData, tmpPitch, dims);

	allocateProjectionData(D_projData, projPitch, dims);
	zeroProjectionData(D_projData, projPitch, dims);

	// We can't precompute pixelWeights when using a volume mask
#if 0 
	if (!useVolumeMask)
#endif
		precomputeWeights();

	// TODO: check if allocations succeeded
	return true;
}

bool EM::precomputeWeights()
{
	zeroVolumeData(D_pixelWeight, pixelPitch, dims);
#if 0
	if (useSinogramMask) {
		callBP(D_pixelWeight, pixelPitch, D_smaskData, smaskPitch);
	} else
#endif
	{
		processSino<opSet>(D_projData, 1.0f, projPitch, dims);
		callBP(D_pixelWeight, pixelPitch, D_projData, projPitch, 1.0f);
	}
	processVol<opInvert>(D_pixelWeight, pixelPitch, dims);

#if 0
	if (useVolumeMask) {
		// scale pixel weights with mask to zero out masked pixels
		processVol<opMul>(D_pixelWeight, D_maskData, pixelPitch, dims);
	}
#endif

	return true;
}

bool EM::iterate(unsigned int iterations)
{
#if 0
	if (useVolumeMask)
		precomputeWeights();
#endif

	// iteration
	for (unsigned int iter = 0; iter < iterations && !astra::shouldAbort(); ++iter) {

		// Do FP of volumeData 
		zeroProjectionData(D_projData, projPitch, dims);
		callFP(D_volumeData, volumePitch, D_projData, projPitch, 1.0f);

		// Divide sinogram by FP (into projData)
		processSino<opDividedBy>(D_projData, D_sinoData, projPitch, dims);

		// Do BP of projData into tmpData
		zeroVolumeData(D_tmpData, tmpPitch, dims);
		callBP(D_tmpData, tmpPitch, D_projData, projPitch, 1.0f);

		// Multiply volumeData with tmpData divided by pixel weights
		processVol<opMul2>(D_volumeData, D_tmpData, D_pixelWeight, pixelPitch, dims);

	}

	return true;
}

float EM::computeDiffNorm()
{
	// copy sinogram to projection data
	duplicateProjectionData(D_projData, D_sinoData, sinoPitch, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_tmpData, D_volumeData, volumePitch, dims);
			processVol<opMul>(D_tmpData, D_maskData, tmpPitch, dims);
			callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
	}


	// compute norm of D_projData

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	return sqrt(s);
}


}
