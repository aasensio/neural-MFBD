from configobj import ConfigObj

__all__ = ['Config']

def _lower_to_sep(string, separator='='):
    line=string.partition(separator)
    string=str(line[0]).lower()+str(line[1])+str(line[2])
    return string

class Config(object):

    def __init__(self, configuration_file=None, training=True):
        self.configuration_file = configuration_file

        if (self.configuration_file is None):
            print("Simulator initialized without configuration file. Use `read_configuration` to read a configuration file.")
        else:
            self.read_configuration(self.configuration_file)

        self.hyperparameters = {}

        if (training):
            # Training
            self.hyperparameters['lr'] = float(self.config_dict['training']['learning rate'])
            self.hyperparameters['wd'] = float(self.config_dict['training']['weight decay'])
            self.hyperparameters['gpus'] = [int(f) for f in self.config_dict['training']['gpus']]
            self.hyperparameters['batch_size'] = int(self.config_dict['training']['batch size'])
            self.hyperparameters['n_epochs'] = int(self.config_dict['training']['epochs'])            
            self.hyperparameters['scheduler_decay'] = float(self.config_dict['training']['scheduler decay'])
            self.hyperparameters['precision'] = self.config_dict['training']['precision']            
            self.hyperparameters['frequency_png'] = int(self.config_dict['training']['frequency png'])
            self.hyperparameters['gamma_modes'] = float(self.config_dict['training']['regularization parameter for modes'])            
            
            # Images
            self.hyperparameters['npix_apodization'] = int(self.config_dict['images']['apodization in pixel'])
            self.hyperparameters['n_modes'] = int(self.config_dict['images']['number of modes'])
            self.hyperparameters['basis_for_wavefront'] = self.config_dict['images']['basis for wavefront']
            self.hyperparameters['n_frames'] = int(self.config_dict['images']['number of frames'])
            self.hyperparameters['n_pixel'] = int(self.config_dict['images']['number of pixel of patches'])
            self.hyperparameters['bands'] = self.config_dict['images']['bands']
            self.hyperparameters['image_filter'] = self.config_dict['images']['image filter']

            # Telescope
            self.hyperparameters['wavelengths'] = [float(f) for f in self.config_dict['telescope']['wavelengths']]
            self.hyperparameters['pix_size'] = [float(f) for f in self.config_dict['telescope']['pixel size']]
            self.hyperparameters['diameter'] = [float(f) for f in self.config_dict['telescope']['diameter']]
            self.hyperparameters['central_obs'] = [float(f) for f in self.config_dict['telescope']['central obscuration']]            
                                    
            # Dataset                
            self.hyperparameters['dataset_instrument'] = self.config_dict['dataset']['instrument']
            self.hyperparameters['training_file'] = self.config_dict['dataset']['training set']
            self.hyperparameters['validation_split'] = float(self.config_dict['dataset']['validation split'])                                    

            if (self.hyperparameters['dataset_instrument'] == 'HiFi'):
                self.hyperparameters['n_patches_per_image'] = int(self.config_dict['hifi']['number of patches per image'])
                self.hyperparameters['border_pixel'] = int(self.config_dict['hifi']['pixels to avoid on border'])
                

            # Architecture
            self.hyperparameters['resnet_type'] = self.config_dict['architecture']['resnet type']
            self.hyperparameters['n_internal_channels'] = int(self.config_dict['architecture']['internal channels'])
            self.hyperparameters['internal_depth'] = int(self.config_dict['architecture']['internal depth'])
                                                    
    def read_configuration(self, configuration_file):

        f = open(configuration_file, 'r')
        tmp = f.readlines()
        f.close()

        self.configuration_txt = tmp

        input_lower = ['']

        for l in tmp:
            input_lower.append(_lower_to_sep(l)) # Convert keys to lowercase

        self.config_dict = ConfigObj(input_lower)