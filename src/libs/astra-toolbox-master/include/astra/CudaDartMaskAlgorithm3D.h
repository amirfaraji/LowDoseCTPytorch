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

#ifndef _INC_ASTRA_CUDADARTMASKALGORITHM3D
#define _INC_ASTRA_CUDADARTMASKALGORITHM3D

#include "Globals.h"
#include "Config.h"
#include "Algorithm.h"
#include "Float32VolumeData3DMemory.h"


#ifdef ASTRA_CUDA

namespace astra {
	
	class _AstraExport CCudaDartMaskAlgorithm3D : public CAlgorithm
{
	
public:
	
	// type of the algorithm, needed to register with CAlgorithmFactory
	static std::string type;
	
	/** Default constructor, containing no code.
	 */
	CCudaDartMaskAlgorithm3D();
	
	/** Destructor.
	 */
	virtual ~CCudaDartMaskAlgorithm3D();

	/** Initialize the algorithm with a config object.
	 *
	 * @param _cfg Configuration Object
	 * @return initialization successful?
	 */
	virtual bool initialize(const Config& _cfg);

	/** Initialize class, use sequential order.
	 *
	 * @param _pSegmentation		...
	 * @param iConn					...
	 */
	//bool initialize(CFloat32VolumeData2D* _pSegmentation, int _iConn);

	/** Get all information parameters
	 *
	 * @return map with all boost::any object
	 */
	virtual std::map<std::string,boost::any> getInformation();

	/** Get a single piece of information represented as a boost::any
	 *
	 * @param _sIdentifier identifier string to specify which piece of information you want
	 * @return boost::any object
	 */
	virtual boost::any getInformation(std::string _sIdentifier);

	/** Get a description of the class.
	 *
	 * @return description string
	 */
	virtual std::string description() const;

	/** Perform a number of iterations.
	 *
	 * @param _iNrIterations amount of iterations to perform.
	 */
	virtual void run(int _iNrIterations = 0);


protected:
	/** Check this object.
	 *
	 * @return object initialized
	 */
	bool _check();

	unsigned int m_iConn;
	unsigned int m_iThreshold;
	unsigned int m_iRadius;
	int m_iGPUIndex;

	CFloat32VolumeData3DMemory* m_pSegmentation;
	CFloat32VolumeData3DMemory* m_pMask;

};

// inline functions
inline std::string CCudaDartMaskAlgorithm3D::description() const { return CCudaDartMaskAlgorithm3D::type; };

} // end namespace

#endif // ASTRA_CUDA

#endif 
