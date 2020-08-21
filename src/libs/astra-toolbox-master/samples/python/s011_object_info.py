# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra

# Create two volume geometries
vol_geom1 = astra.create_vol_geom(256, 256)
vol_geom2 = astra.create_vol_geom(512, 256)

# Create volumes
v0 = astra.data2d.create('-vol', vol_geom1)
v1 = astra.data2d.create('-vol', vol_geom2)
v2 = astra.data2d.create('-vol', vol_geom2)

# Show the currently allocated volumes
astra.data2d.info()


astra.data2d.delete(v2)
astra.data2d.info()

astra.data2d.clear()
astra.data2d.info()



# The same clear and info command also work for other object types:
astra.algorithm.info()  
astra.data3d.info()
astra.projector.info()
astra.matrix.info()
