import odl
import numpy as np

class RayTransform():

    def __init__(self, 
                min_pt = [-0.13, -0.13], 
                max_pt = [0.13, 0.13], 
                image_shape = (1000, 1000), 
                recon_image_shape = (362, 362),
                detector_shape = (513,),
                num_angles = 1000,
                impl = 'astra_cpu',
            ):
        
        self._MIN_PT = min_pt
        self._MAX_PT = max_pt

        self._IMAGE_SHAPE = image_shape 
        self._RECO_IM_SHAPE = recon_image_shape
        self._DETECTOR_SHAPE = detector_shape
        self._NUM_ANGLES = num_angles

        self.impl = impl

        self.create_ray_transform()
        self.create_recon_ray_transform()
        self.create_fbp_op()

    def create_ray_transform(self):

        space = odl.uniform_discr(min_pt=self._MIN_PT, max_pt=self._MAX_PT, shape=self._IMAGE_SHAPE, dtype=np.float64)
        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=self._NUM_ANGLES, det_shape=self._DETECTOR_SHAPE)

        self.ray_trafo = odl.tomo.RayTransform(space, geometry, impl=self.impl)

    def create_recon_ray_transform(self):

        space = odl.uniform_discr(min_pt=self._MIN_PT, max_pt=self._MAX_PT, shape=self._RECO_IM_SHAPE, dtype=np.float64)
        geometry = odl.tomo.parallel_beam_geometry(space, num_angles=self._NUM_ANGLES, det_shape=self._DETECTOR_SHAPE)

        self.recon_ray_trafo = odl.tomo.RayTransform(space, geometry, impl=self.impl)

    def create_fbp_op(self):

        self.fbp = odl.tomo.fbp_op(self.recon_ray_trafo,
            filter_type='Hann',
            frequency_scaling=1)