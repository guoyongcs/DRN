import os
from data import srdata


class DF2K(srdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        super(DF2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(DF2K, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'DF2K_HR')
        self.dir_lr = os.path.join(self.apath, 'DF2K_LR_bicubic')

