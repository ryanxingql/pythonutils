class BaseAlg():
    def __init__(self):
        super().__init__()

    def create_model(self):
        pass

    def print_net(self):
        pass

    def create_optimizer(self):
        pass

    def save_model(self):
        pass

# ===
# Scheduler
# ===

# MultiStepRestartLR
# CosineAnnealingRestartLR
# https://github.com/RyanXingQL/SubjectiveQE-ESRGAN/blob/main/utils/deep_learning.py
