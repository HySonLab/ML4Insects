from utils.bash_utils import get_args
from EPGS.trainer import Trainer
from utils.configs_utils import process_config
from utils.train_utils import get_architecture, get_dataset_group
from models import *

# Define a utility function to get the file names
# config = process_config('examples.json')
def main():
    try:
        args = get_args()
        config = process_config(args.config)
        model = get_architecture(config.arch)
    except Exception:
        print("missing or invalid arguments")
        exit()
    trainer = Trainer(model,config)
    all_dataset_names = get_dataset_group(trainer.config.dataset_name) 
    # Get the list of all folders corresponding to dataset_name. 
    trainer.generate_data(all_dataset_names)
    trainer.train()
    trainer.test()
    
    trainer.write_log()
    trainer.plot_result(savefig = True)
    trainer.save_checkpoint()

if __name__ == '__main__':
    main()