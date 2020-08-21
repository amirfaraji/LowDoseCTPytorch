from progress.bar import Bar

class CustomBar(Bar):
    suffix='%(index)d / %(max)d - %(eta)d secs - loss: %(loss).2f, ssim: %(ssim).2f, psnr: %(psnr).2f'

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, val):
        self.__loss = val
    
    @property
    def ssim(self):
        return self.__ssim

    @ssim.setter
    def ssim(self, val):
        self.__ssim = val
    
    @property
    def psnr(self):
        return self.__psnr

    @psnr.setter
    def psnr(self, val):
        self.__psnr = val