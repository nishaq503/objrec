from .plugin import Plugin
from collections import OrderedDict


class LearningRateScheduler(Plugin):
    def __init__(self, learning_rate_schedule: callable, verbose=False):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate_schedule = learning_rate_schedule
        self.learning_rate_by_epoch = OrderedDict()
        self.optimizer = None
        self.verbose = verbose

    def register(self, trainer):
        self.optimizer = trainer.optimizer
        trainer.events.pre_epoch.append(self.pre_epoch_handler)

    @staticmethod
    def set_optimizer_to_new_lr(lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def pre_epoch_handler(self, **kwargs):
        if kwargs['epoch_count'] == 1:
            return

        optimizer = kwargs['optimizer']

        new_learning_rate = self.learning_rate_schedule(self, **kwargs)

        if new_learning_rate is not None:
            self.set_optimizer_to_new_lr(new_learning_rate, optimizer)
            self.learning_rate_by_epoch[kwargs['epoch_count']] = new_learning_rate

            if self.verbose:
                print('lr -> {}'.format(new_learning_rate))

    def __str__(self):
        return 'lr: {}'.format(self.learning_rate_by_epoch[reversed(self.learning_rate_by_epoch).__next__()])