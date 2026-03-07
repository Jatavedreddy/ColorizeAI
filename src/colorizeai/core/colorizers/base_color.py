
import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm # this will make the l channel roughly in the range [-0.5, 0.5]

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm # this will make the ab channels roughly in the range [-0.5, 0.5]

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm

