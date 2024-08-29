
def main(lr, warm_lr, batch_size, warm_epoch, n_epoch):
    # If want to use this method to train model, path in the configuration files need to be changed.
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

#
# def optimize_bayes():
#     pbounds = {
#         'SOLVER.BASE_LR': (1e-4, 1e-2),
#         'SOLVER.WARMUP_START_LR': (1e-5, 1e-3),
#         'SOLVER.WARMUP_EPOCHS': (0.0, 10.0),
#         'SOLVER.MAX_EPOCH': (15, 50)
#     }
#
#     # Define the function to be optimized
#     def wrapped_main(**params):
#         # Extract parameters from kwargs
#         lr = params['SOLVER.BASE_LR']
#         warm_lr = params['SOLVER.WARMUP_START_LR']
#         batch_size = int(params['TRAIN.BATCH_SIZE'])
#         warm_epoch = int(params['SOLVER.WARMUP_EPOCHS'])
#         n_epoch = int(params['SOLVER.MAX_EPOCH'])
#
#         # Call the main function with these parameters
#         return main(lr, warm_lr, batch_size, warm_epoch, n_epoch)
#
#     # Initialize the optimizer
#     optimizer = BayesianOptimization(
#         f=wrapped_main,
#         pbounds=pbounds,
#         random_state=1
#     )
#
#     # Initialize tqdm progress bar
#     with tqdm(total=30) as pbar:
#         def tqdm_callback(optim_result):
#             pbar.update(1)
#
#         # Execute optimization with callback for progress
#         optimizer.maximize(
#             init_points=5,  # Initial random search
#             n_iter=20,      # Number of iterations
#             acq='ei',       # Acquisition function
#             callback=tqdm_callback
#         )
#
#     # Print the best result
#     print("Best parameters found:")
#     print(optimizer.max)


if __name__ == '__main__':
    main(1e-3, 1e-4, 32, 5.0, 0)

