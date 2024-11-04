import os

def base_path_creator(core_args, create=True):
    path = '.'
    path = next_dir(path, 'experiments', create=create)
    path = next_dir(path, '_'.join(core_args.data_name), create=create)
    path = next_dir(path, '_'.join(core_args.model_name), create=create)
    path = next_dir(path, core_args.task, create=create)
    path = next_dir(path, core_args.language, create=create)
    if core_args.seed != 1:
        path = next_dir(path, f'seed{core_args.seed}', create=create)
    return path

def create_attack_base_path(attack_args, path='.', mode='train', create=True):
    # Choose the base directory based on mode
    base_dir = 'attack_train' if mode == 'train' else 'attack_eval'
    path = next_dir(path, base_dir, create=create)

    # Common directory structure for both train and eval
    path = next_dir(path, attack_args.attack_method, create=create)
    if attack_args.attack_command != 'mute':
        path = next_dir(path, f'command_{attack_args.attack_command}', create=create)
    if attack_args.attack_token != 'eot':
        path = next_dir(path, f'attack_token{attack_args.attack_token}', create=create)
    if attack_args.attack_init != 'random':
        attack_init_path_str = attack_args.attack_init
        attack_init_path_str = '-'.join(attack_init_path_str.split('/'))
        path = next_dir(path, f'attack_init_{attack_init_path_str}', create=create)
    path = next_dir(path, f'attack_size{attack_args.attack_size}', create=create)
    path = next_dir(path, f'clip_val{attack_args.clip_val}', create=create)

    # Additional directory for eval mode
    if mode == 'eval':
        path = next_dir(path, f'attack-epoch{attack_args.attack_epoch}', create=create)
    
    return path

def attack_base_path_creator_train(attack_args, path='.', create=True):
    return create_attack_base_path(attack_args, path, 'train', create)

def attack_base_path_creator_eval(attack_args, path='.', create=True):
    return create_attack_base_path(attack_args, path, 'eval', create)



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