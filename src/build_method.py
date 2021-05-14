def build_method(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.local_update == 'base':
        from local_update_set.update_only_fisher import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'ewc':
        from local_update_set.ewc import LocalUpdate as LocalUpdateModule
    
    if args.global_update == 'avg':
        from aggregate_utils import average_weights as GlobalUpdate
    elif args.global_update == 'weighted_avg':
        from aggregate_utils import average_weights_with_fisher as GlobalUpdate

    return LocalUpdateModule, GlobalUpdate
    
