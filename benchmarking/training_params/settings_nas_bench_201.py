# Trainings parameters used for NAS-Bench-201 training
optimizer = "SGD"
Nesterov = True
momentum = 0.9
weight_decay = 0.0005
batch_size = 256
V = 4
random_flip_p = 0.5
normalization = True

initial_LR = 0.1
ending_LR = 0
LR_schedule = "cosine"
epochs = 200
initial_channel = 16
N = 5
random_crop = True
