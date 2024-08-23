import os
import optuna

from bayes_opt import BayesianOptimization
from tqdm import tqdm

from helper.configurations import get_raw_data_dir, get_clip_data_dir, get_frame_data_dir
from helper.data_pre_process import (
    dataset_partition,
    dataset_get_vis,
)
from helper.dataset_helper import (
    generate_combined_labels
)
from helper.video_helper import (
    combine_videos, combine_frames
)


def generate_dataset():
    """
    Start the data sampling and partition process after the video splitting is done.
    This method will generate the train/test/val sets for each week and for combined 4 weeks.
    """

    yyyy = "2022"
    ww = "w1"

    data_dir, video_dir, label_dir = get_frame_data_dir(yyyy, ww)

    train_labels = []
    test_labels = []
    val_labels = []

    output_dir = os.path.join(data_dir, "../csv/wall")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in range(1, 5):
        yyyy = "2022"
        ww = f"w{i}"

        data_dir_k, video_dir, label_dir = get_frame_data_dir(yyyy, ww)

        dataset_partition(
            video_dir, label_dir, data_dir_k,
            target_labels=["UserAwkwardness"],
            sampling="oversample",
            train_scales=0.7,
            test_scales=0.15,
            val_scales=0.15
        )

        print(f"Dataset partition is done for: week {i}. ")

        dataset_get_vis(
            data_dir_k,
            video_dir,
            label_dir,
        )

        train_labels.append(os.path.join(data_dir_k, "train.csv"))
        test_labels.append(os.path.join(data_dir_k, "test.csv"))
        val_labels.append(os.path.join(data_dir_k, "val.csv"))

    generate_combined_labels(train_labels, os.path.join(output_dir, "train.csv"))
    generate_combined_labels(test_labels, os.path.join(output_dir, "test.csv"))
    generate_combined_labels(val_labels, os.path.join(output_dir, "val.csv"))

    return 0


def main(lr, warm_lr, batch_size, warm_epoch, n_epoch):
    import subprocess

    cfg_path = 'config/ua_2022_slowfast.yaml'

    command = [
        'python', '../slowfast/tools/run_net.py',
        '--cfg', str(cfg_path),
        'TRAIN.EVAL_PERIOD', str(1),
        'SOLVER.BASE_LR', str(lr),
        'SOLVER.WARMUP_START_LR', str(warm_lr),
        'TRAIN.BATCH_SIZE', str(batch_size),
        'SOLVER.MAX_EPOCH', str(n_epoch),
        'SOLVER.WARMUP_EPOCHS', str(warm_epoch),
        'TENSORBOARD.CLASS_NAMES_PATH', f"config/names/class_id_ua.json",
    ]

    print(command)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print("Command output:")
    # print(result.stdout)
    # print("Command error output:")
    # print(result.stderr)

    for line in result.stdout.split('\n'):
        if "Macro accuracies: " in line:
            accuracy = float(line.split(":")[-1].strip())
            return accuracy

    return 0.0


def optimize_bayes():
    pbounds = {
        'SOLVER.BASE_LR': (1e-4, 1e-2),
        'SOLVER.WARMUP_START_LR': (1e-5, 1e-3),
        'SOLVER.WARMUP_EPOCHS': (0.0, 10.0),
        'SOLVER.MAX_EPOCH': (15, 50)
    }

    # Define the function to be optimized
    def wrapped_main(**params):
        # Extract parameters from kwargs
        lr = params['SOLVER.BASE_LR']
        warm_lr = params['SOLVER.WARMUP_START_LR']
        batch_size = int(params['TRAIN.BATCH_SIZE'])
        warm_epoch = int(params['SOLVER.WARMUP_EPOCHS'])
        n_epoch = int(params['SOLVER.MAX_EPOCH'])

        # Call the main function with these parameters
        return main(lr, warm_lr, batch_size, warm_epoch, n_epoch)

    # Initialize the optimizer
    optimizer = BayesianOptimization(
        f=wrapped_main,
        pbounds=pbounds,
        random_state=1
    )

    # Initialize tqdm progress bar
    with tqdm(total=30) as pbar:
        def tqdm_callback(optim_result):
            pbar.update(1)

        # Execute optimization with callback for progress
        optimizer.maximize(
            init_points=5,  # Initial random search
            n_iter=20,      # Number of iterations
            acq='ei',       # Acquisition function
            callback=tqdm_callback
        )

    # Print the best result
    print("Best parameters found:")
    print(optimizer.max)


def optimize_optuna():
    # Define the function to be optimized
    def wrapped_main(trail):
        lr = trail.suggest_loguniform('SOLVER.BASE_LR', 1e-4, 1e-2)
        warm_lr = trail.suggest_loguniform('SOLVER.WARMUP_START_LR', 1e-5, 1e-3)
        batch_size = 32 # trail.suggest_int('TRAIN.BATCH_SIZE', 16, 32)
        warm_epoch = trail.suggest_discrete_uniform('SOLVER.WARMUP_EPOCHS', 0.0, 10.0, 1.0)
        n_epoch = trail.suggest_int('SOLVER.MAX_EPOCH', 20, 40)

        # Call the main function with these parameters
        return main(lr, warm_lr, batch_size, warm_epoch, n_epoch)

    # Initialize the optimizer
    study = optuna.create_study()
    study.optimize(wrapped_main, n_trials=1)

    result = main(1e-3, 1e-4, 32, 5.0, 25)

    return 0


if __name__ == '__main__':
    generate_dataset()
    # main(1e-3, 1e-4, 32, 5.0, 0)

