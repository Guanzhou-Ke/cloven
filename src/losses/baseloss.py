class BaseLoss:
    """
    Base Loss
    """
    def __init__(self, args, device='cpu') -> None:
        self.args = args
        self.device = device
        
    def __call__(self, *args, **kwds):
        return self.get_loss(*args, **kwds)
    
    def get_loss(self, *args, **kwds):
        # print(args, kwds)
        return None