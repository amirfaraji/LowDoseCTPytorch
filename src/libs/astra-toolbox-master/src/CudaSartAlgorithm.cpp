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

#ifdef ASTRA_CUDA

#include "astra/CudaSartAlgorithm.h"

#include "astra/cuda/2d/sart.h"

using namespace std;

namespace astra {

// type of the algorithm, needed to register with CAlgorithmFactory
std::string CCudaSartAlgorithm::type = "SART_CUDA";

//----------------------------------------------------------------------------------------
// Constructor
CCudaSartAlgorithm::CCudaSartAlgorithm() 
{
	m_bIsInitialized = false;
	CCudaReconstructionAlgorithm2D::_clear();
}

//----------------------------------------------------------------------------------------
// Destructor
CCudaSartAlgorithm::~CCudaSartAlgorithm() 
{
	// The actual work is done by ~CCudaReconstructionAlgorithm2D
}

//---------------------------------------------------------------------------------------
// Initialize - Config
bool CCudaSartAlgorithm::initialize(const Config& _cfg)
{
	ASTRA_ASSERT(_cfg.self);
	ConfigStackCheck<CAlgorithm> CC("CudaSartAlgorithm", this, _cfg);

	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_cfg);

	if (!m_bIsInitialized)
		return false;

	astraCUDA::SART *sart = new astraCUDA::SART();

	m_pAlgo = sart;
	m_bAlgoInit = false;

	// projection order
	int projectionCount = m_pSinogram->getGeometry()->getProjectionAngleCount();
	int* projectionOrder = NULL;
	string projOrder = _cfg.self.getOption("ProjectionOrder", "random");
	CC.markOptionParsed("ProjectionOrder");
	if (projOrder == "sequential") {
		projectionOrder = new int[projectionCount];
		for (int i = 0; i < projectionCount; i++) {
			projectionOrder[i] = i;
		}
		sart->setProjectionOrder(projectionOrder, projectionCount);
		delete[] projectionOrder;
	} else if (projOrder == "random") {
		projectionOrder = new int[projectionCount];
		for (int i = 0; i < projectionCount; i++) {
			projectionOrder[i] = i;
		}
		for (int i = 0; i < projectionCount-1; i++) {
			int k = (rand() % (projectionCount - i));
			int t = projectionOrder[i];
			projectionOrder[i] = projectionOrder[i + k];
			projectionOrder[i + k] = t;
		}
		sart->setProjectionOrder(projectionOrder, projectionCount);
		delete[] projectionOrder;
	} else if (projOrder == "custom") {
		vector<float32> projOrderList = _cfg.self.getOptionNumericalArray("ProjectionOrderList");
		projectionOrder = new int[projOrderList.size()];
		for (unsigned int i = 0; i < projOrderList.size(); i++) {
			projectionOrder[i] = static_cast<int>(projOrderList[i]);
		}
		sart->setProjectionOrder(projectionOrder, projectionCount);
		delete[] projectionOrder;
		CC.markOptionParsed("ProjectionOrderList");
	}

	m_fLambda = _cfg.self.getOptionNumerical("Relaxation", 1.0f);
	CC.markOptionParsed("Relaxation");

	return true;
}

//---------------------------------------------------------------------------------------
// Initialize - C++
bool CCudaSartAlgorithm::initialize(CProjector2D* _pProjector,
                                     CFloat32ProjectionData2D* _pSinogram, 
                                     CFloat32VolumeData2D* _pReconstruction)
{
	m_bIsInitialized = CCudaReconstructionAlgorithm2D::initialize(_pProjector, _pSinogram, _pReconstruction);

	if (!m_bIsInitialized)
		return false;

	m_fLambda = 1.0f;

	m_pAlgo = new astraCUDA::SART();
	m_bAlgoInit = false;

	return true;
}

//----------------------------------------------------------------------------------------

void CCudaSartAlgorithm::initCUDAAlgorithm()
{
	CCudaReconstructionAlgorithm2D::initCUDAAlgorithm();

	astraCUDA::SART* pSart = dynamic_cast<astraCUDA::SART*>(m_pAlgo);

	pSart->setRelaxation(m_fLambda);
}



} // namespace astra

#endif // ASTRA_CUDA
