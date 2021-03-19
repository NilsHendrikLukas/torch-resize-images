import os
import GPUtil as GPUtil


def pick_gpu():
    """
    Picks a GPU with the least memory load.
    :return:
    """
    try:
        gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=2, maxMemory=0.8, includeNan=False,
                                       excludeID=[], excludeUUID=[])[0]
        return gpu
    except Exception as e:
        print(e)
        return "0"


def reserve_gpu(mode_or_id):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if mode_or_id:
        gpu_id = mode_or_id
        os.environ["CUDA_VISIBLE_DEVICES"] = mode_or_id
    else:
        gpu_id = str(pick_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Selecting GPU id {gpu_id}")
