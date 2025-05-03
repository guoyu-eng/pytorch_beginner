from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


wirter = SummaryWriter('logs')

for i in range(10):
    wirter.add_scalar('y=x', i, i)
# wirter.add_image()

wirter.close()