from .cross_scan import cross_selective_scan


def install_depthart(module):
    """Install the mixed-precision scan into an imported DepthART tvimblock.

    This changes only the module-global function resolved by SS2D.forward_core;
    model classes and checkpoints remain untouched.
    """
    if not hasattr(module, "cross_selective_scan"):
        raise TypeError("module does not expose DepthART cross_selective_scan")
    previous = module.cross_selective_scan
    module.cross_selective_scan = cross_selective_scan
    return previous
