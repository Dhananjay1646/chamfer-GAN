from torch import nn
import torch.nn.functional as F
import torch
import os
import argparse

##########################################################
## init stuff and arguments
os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
#---------------------------------------------------------

##########################################################
cuda = True if torch.cuda.is_available() else False
#---------------------------------------------------------

##########################################################
## init weights function
def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)
#---------------------------------------------------------

#######################################################################################
## generator class
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.init_size = opt.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
			nn.Tanh(),
		)

	def forward(self, z):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks(out)
		return img
#---------------------------------------------------------------------------------------

########################################################################################
## loss function (needs to be defined)
loss_function = None#?
#---------------------------------------------------------------------------------------

########################################################################################
## init generator
generator = Generator()
if cuda:
	generator.cuda()
#---------------------------------------------------------------------------------------

########################################################################################
## init weights
generator.apply(weights_init_normal)
#---------------------------------------------------------------------------------------

##########################################################################################
# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
	datasets.MNIST(
		"data/mnist",
		train=True,
		download=True,
		transform=transforms.Compose(
			[transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
		),
	),
	batch_size=opt.batch_size,
	shuffle=True,
)
#----------------------------------------------------------------------------------------

##########################################################################################
## define optimizer
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#-----------------------------------------------------------------------------------------

# define variables
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#############################
##########TRAINING###########
#############################

for epoch in range(opt.n_epochs):
	# the following line should change on how the dataset we want to define
	for 1, (images,_) in enumerate(dataloader):
		# ground truths
		valid_im = Variable(Tensor(images.shape[0], 1).fill_(1.0), requires_grad=False)
		fake_im = Variable(Tensor(images.shape[0], 1).fill_(0.0), requires_grad=False)

		# input images (need to define here)
		real_images = Variable(images.type(Tensor))

		#-----------------------
		# train the generator
		#-----------------------

		generator_optimizer.zero_grad()

		# noise input to generator function
		z = Variable(Tensor(np.random.normal(0, 1, (images.shape[0], opt.latent_dim))))

		# generate images
		generated_images = generator(z)

		# update loss on the generated images (needs to be updated)
		generator_loss = loss_function(generated_images, valid_im)

		generator_loss.backward()
		generator_optimizer.step()

		print(
			"[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
			% (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
			)

		batches_done = epoch * len(dataloader) + i
		if batches_done % opt.sample_interval == 0:
			save_image(generated_images.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
