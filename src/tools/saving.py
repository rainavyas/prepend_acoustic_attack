import os

def base_path_creator(core_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, core_args.data_name, create=create)
    path = next_dir(path, core_args.model_name, create=create)
    path = next_dir(path, core_args.task, create=create)
    path = next_dir(path, core_args.language, create=create)
    if core_args.seed != 1:
        path = next_dir(path, f'seed{core_args.seed}', create=create)
    return path

def attack_base_path_creator_train(attack_args, path='.', create=True):
    path = next_dir(path, 'attack_train', create=create)
    path = next_dir(path, attack_args.attack_method, create=create)
    path = next_dir(path, f'attack_size{attack_args.attack_size}', create=create)
    path = next_dir(path, f'clip_val{attack_args.clip_val}', create=create)
    return path

def attack_base_path_creator_eval(attack_args, path='.', create=True):
    path = next_dir(path, 'attack_eval', create=create)
    path = next_dir(path, attack_args.attack_method, create=create)
    path = next_dir(path, f'attack_size{attack_args.attack_size}', create=create)
    path = next_dir(path, f'clip_val{attack_args.clip_val}', create=create)
    path = next_dir(path, f'attack-epoch{attack_args.attack_epoch}', create=create)
    return path


def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        try:
            if create:
                os.mkdir(f'{path}/{dir_name}')
            else:
                raise ValueError ("provided args do not give a valid model path")
        except:
            # path has already been created in parallel
            pass
    path += f'/{dir_name}'
    return path