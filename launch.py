from utils.bash_utils import get_args
from EPGS.trainer import cross_validate
from utils.configs_utils import process_config
from utils.train_utils import get_architecture
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
    
    cv = cross_validate(model, config, n_folds = 10)
    cv.CV()
    cv.write_log()
    cv.plot_summary(savefig = True)
    cv.save_best_model()

if __name__ == '__main__':
    main()
