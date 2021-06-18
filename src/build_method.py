def build_method(args):
    #if args.local_update == 'base':
    #    from local_update_set.base import LocalUpdate as LocalUpdateModule
    if args.local_update == 'base':
        from local_update_set.update_only_fisher import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'ewc':
        from local_update_set.ewc import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'rwalk':
        from local_update_set.rwalk import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'rwalk_v2':
        from local_update_set.rwalk_v2 import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'rwalk_v3':
        from local_update_set.rwalk_v3 import LocalUpdate as LocalUpdateModule
    elif args.local_update == 'l2':
        from local_update_set.l2 import LocalUpdate as LocalUpdateModule
    
    if args.global_update == 'avg':
        from aggregate_utils import average_weights as GlobalUpdate
    elif args.global_update == 'weighted_avg':
        from aggregate_utils import average_weights_with_fisher as GlobalUpdate

    elif args.global_update == 'normalized_weighted_avg':
        from aggregate_utils import average_weights_with_fisher_normalized as GlobalUpdate

    return LocalUpdateModule, GlobalUpdate
    
