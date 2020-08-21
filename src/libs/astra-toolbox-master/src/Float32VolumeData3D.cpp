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

#include "astra/Float32VolumeData3D.h"

namespace astra
{

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32VolumeData3D::CFloat32VolumeData3D() :
	CFloat32Data3D() {

}

//----------------------------------------------------------------------------------------
// Destructor
CFloat32VolumeData3D::~CFloat32VolumeData3D() {

}

void CFloat32VolumeData3D::changeGeometry(CVolumeGeometry3D* _pGeometry)
{
	if (!m_bInitialized) return;

	delete m_pGeometry;
	m_pGeometry = _pGeometry->clone();
}


} // end namespace astra
