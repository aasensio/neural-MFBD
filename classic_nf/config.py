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
            self.hyperparameters['lr'] = float(self.config_dict['training']['learning rate'])
            self.hyperparameters['wd'] = float(self.config_dict['training']['weight decay'])
            self.hyperparameters['batch_size'] = int(self.config_dict['training']['batch size'])
            self.hyperparameters['n_epochs'] = int(self.config_dict['training']['epochs'])
            self.hyperparameters['gpus'] = [int(f) for f in self.config_dict['training']['gpus']]
            self.hyperparameters['scheduler_decay'] = float(self.config_dict['training']['scheduler decay'])
            self.hyperparameters['training_file'] = self.config_dict['dataset']['training set']
            self.hyperparameters['validation_split'] = float(self.config_dict['dataset']['validation split'])
            self.hyperparameters['pix_size'] = float(self.config_dict['telescope']['pixel size'])
            self.hyperparameters['diameter'] = float(self.config_dict['telescope']['diameter'])
            self.hyperparameters['central_obs'] = float(self.config_dict['telescope']['central obscuration'])
            self.hyperparameters['wavelength'] = float(self.config_dict['telescope']['wavelength'])            
            self.hyperparameters['precision'] = self.config_dict['training']['precision']            
            self.hyperparameters['npix_apodization'] = int(self.config_dict['images']['apodization in pixel'])
            self.hyperparameters['n_modes'] = int(self.config_dict['images']['number of modes'])
            self.hyperparameters['frequency_png'] = int(self.config_dict['training']['frequency png'])
            self.hyperparameters['object_enc_channels'] = int(self.config_dict['architecture']['objectnet encoder channels'])
            self.hyperparameters['object_enc_channels_out'] = int(self.config_dict['architecture']['objectnet encoder feature channels'])
            self.hyperparameters['object_channels_proj'] = int(self.config_dict['architecture']['objectnet feature projection channels'])
            self.hyperparameters['object_channels_weight'] = int(self.config_dict['architecture']['objectnet feature weight channels'])
            self.hyperparameters['gamma_modes'] = float(self.config_dict['training']['regularization parameter for modes'])            
            self.hyperparameters['basis_for_wavefront'] = self.config_dict['images']['basis for wavefront']
            self.hyperparameters['n_frames'] = int(self.config_dict['images']['number of frames'])
            self.hyperparameters['n_training_per_image'] = int(self.config_dict['dataset']['number of bursts per image'])
            self.hyperparameters['n_pixel_patch'] = int(self.config_dict['images']['number of pixel of patches'])
            self.hyperparameters['n_pixel'] = int(self.config_dict['images']['number of pixel of images'])
            self.hyperparameters['border_pixel'] = int(self.config_dict['images']['border of observations'])
            self.hyperparameters['gradient_steps'] = int(self.config_dict['training']['number of gradient steps'])
            self.hyperparameters['type_cleaning'] = self.config_dict['training']['type of cleaning']
            # self.hyperparameters['gamma_obj'] = [float(f) for f in self.config_dict['training']['gamma object']]
            
        else:            
            self.batch_size = int(self.config_dict['validation']['batch size'])
            self.gpus = [int(f) for f in self.config_dict['validation']['gpus']]
            self.npix_apodization = int(self.config_dict['images']['apodization in pixel'])
            self.validation_file = self.config_dict['dataset']['validation set']        
            self.zero_average_tiptilt = True if self.config_dict['validation']['zero average tiptilt'] == 'True' else False
            self.number_of_modes = int(self.config_dict['images']['number of modes'])
            self.npix_border_loss = int(self.config_dict['images']['border to remove from loss in pixel'])
            self.checkpoint = self.config_dict['validation']['checkpoint']            
                
    def read_configuration(self, configuration_file):

        f = open(configuration_file, 'r')
        tmp = f.readlines()
        f.close()

        self.configuration_txt = tmp

        input_lower = ['']

        for l in tmp:
            input_lower.append(_lower_to_sep(l)) # Convert keys to lowercase

        self.config_dict = ConfigObj(input_lower)