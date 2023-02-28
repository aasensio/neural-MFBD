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
            self.hyperparameters['frequency_png'] = int(self.config_dict['training']['frequency png'])
            self.hyperparameters['n_pixel'] = int(self.config_dict['training']['number of pixels'])
            
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