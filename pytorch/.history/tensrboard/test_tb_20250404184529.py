from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


wirter = SummaryWriter('logs')

for i in range(100):
    wirter.add_scalar('y=2x', 3*i, i)
# wirter.add_image()
# tensorboard --logdir=logs 
#  -- port=6006
# the tensorboad the graph log files
wirter.close()