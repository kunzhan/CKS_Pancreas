import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg import VGG16

class unetUp(nn.Module):
	def __init__(self, in_size, out_size):
		super(unetUp, self).__init__()
		self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, inputs1, inputs2):
		mid = F.interpolate(inputs2,(inputs1.shape[2],inputs1.shape[3]), mode='bilinear', align_corners=True)
		outputs = torch.cat([inputs1, mid], 1)
		outputs = self.conv1(outputs)
		outputs = self.relu(outputs)
		outputs = self.conv2(outputs)
		outputs = self.relu(outputs)
		return outputs

class Unet(nn.Module):
	def __init__(self, num_classes=3, in_channels=3, pretrained=False):
		super(Unet, self).__init__()
		self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)
		in_filters = [192, 384, 768, 1024]
		out_filters = [64, 128, 256, 512]
		self.up_concat4 = unetUp(in_filters[3], out_filters[3])
		self.up_concat3 = unetUp(in_filters[2], out_filters[2])
		self.up_concat2 = unetUp(in_filters[1], out_filters[1])
		self.up_concat1 = unetUp(in_filters[0], out_filters[0])
		self.final = nn.Conv2d(out_filters[0], num_classes, 1)

	def forward(self, inputs):
		feat1 = self.vgg.features[  :4 ](inputs)
		feat2 = self.vgg.features[4 :9 ](feat1)
		feat3 = self.vgg.features[9 :16](feat2)
		feat4 = self.vgg.features[16:23](feat3)
		feat5 = self.vgg.features[23:-1](feat4)

		up4 = self.up_concat4(feat4, feat5)
		up3 = self.up_concat3(feat3, up4)
		up2 = self.up_concat2(feat2, up3)
		up1 = self.up_concat1(feat1, up2)

		final = self.final(up1)
		
		return final

	def _initialize_weights(self, *stages):
		for modules in stages:
			for module in modules.modules():
				if isinstance(module, nn.Conv2d):
					nn.init.kaiming_normal_(module.weight)
					if module.bias is not None:
						module.bias.data.zero_()
				elif isinstance(module, nn.BatchNorm2d):
					module.weight.data.fill_(1)
					module.bias.data.zero_()

class RSTN(nn.Module):
	def __init__(self, crop_margin=25, crop_prob=0.5, \
					crop_sample_batch=1, n_class=3, TEST=None):
		super(RSTN, self).__init__()
		self.TEST = TEST
		self.margin = crop_margin
		self.prob = crop_prob
		self.batch = crop_sample_batch
		self.coarse_model = Unet()
		self.fine_model = Unet()
		self.fine_model_ema = Unet()
		self._initialize_weights()

	def _initialize_weights(self):
		for name, mod in self.named_children():
			if name == 'saliency1':
				nn.init.xavier_normal_(mod.weight.data)
				mod.bias.data.fill_(1)
			elif name == 'saliency2':
				mod.weight.data.zero_()
				mod.bias.data = torch.tensor([1.0, 1.5, 2.0])
			elif name == 'tt':
				nn.init.xavier_normal_(mod.weight.data)
				mod.bias.data.fill_(1)
			elif name == 'zz':
				nn.init.xavier_normal_(mod.weight.data)
				mod.bias.data.fill_(1)
			elif name == 'oo':
				nn.init.xavier_normal_(mod.weight.data)
				mod.bias.data.fill_(1)

	def forward(self, image, e, label=None, mode=None, score=None, mask=None):
		if self.TEST is None:
			assert label is not None and mode is not None \
				and score is None and mask is None

			if mode == 'S':
				h = image
				h = self.coarse_model(h)
				h = torch.sigmoid(h)
				coarse_prob = h
				cropped_image, crop_info = self.crop(label, image)
				h = cropped_image
				h = self.fine_model(h)
				h = self.uncrop(crop_info, h, image)
				h = torch.sigmoid(h)
				fine_prob = h

				return coarse_prob,fine_prob

			elif mode == 'I':
				if e <= 2:
					h = image
					h = self.coarse_model(h)
					h = torch.sigmoid(h)
					coarse_prob = h
					cropped_image, crop_info = self.crop(coarse_prob, image, label)
					h = cropped_image
					h = self.fine_model(h)
					h = self.uncrop(crop_info, h, image)
					h = torch.sigmoid(h)
					fine_prob = h
					return coarse_prob,fine_prob

				elif e > 2:
					coarse_prob = image*0
					h = image
					h = self.fine_model(h)
					h = torch.sigmoid(h)
					fine_prob = h
					return coarse_prob,fine_prob
			
			elif mode == 'J':
				
				h = image
				h = self.coarse_model(h)
				h = torch.sigmoid(h)
				coarse_prob = h

				cropped_image, crop_info = self.crop(coarse_prob, image, label)

				h = cropped_image
				h = self.fine_model(h)
				h = self.uncrop(crop_info, h, image)
				h = torch.sigmoid(h)
				fine_prob = h
				cropped_image, crop_info = self.crop(label, image)
				h = cropped_image
				h = self.fine_model(h)
				h = self.uncrop(crop_info, h, image)
				h = torch.sigmoid(h)
				fine_prob_gt = h
				return coarse_prob, fine_prob, fine_prob_gt
			

		elif self.TEST == 'C': 
			assert label is None and mode is None and \
				score is None and mask is None
			h = image
			h = self.coarse_model(h)
			h = torch.sigmoid(h)
			coarse_prob = h
			return coarse_prob
		
		elif self.TEST == 'O':
			assert label is not None and mode is None and \
				score is None and mask is None

			cropped_image, crop_info = self.crop(label, image)
			h = cropped_image
			h = self.fine_model(h)
			h = self.uncrop(crop_info, h, image)
			h = torch.sigmoid(h)
			fine_prob = h
			return fine_prob

		elif self.TEST == 'F':
			assert label is None and mode is None \
				and score is not None and mask is not None
			h = score
			cropped_image, crop_info = self.crop(mask, image)
			h = cropped_image
			fine_prob = self.fine_model(h)
			fine_prob = self.uncrop(crop_info, fine_prob, image)			
			fine_prob = torch.sigmoid(fine_prob)
			return fine_prob

		else:
			raise ValueError("wrong value of TEST, should be in [None , 'O']")

	def crop(self, prob_map, saliency_data, label=None):
		(N, C, W, H) = prob_map.shape

		binary_mask = (prob_map >= 0.5) # torch.uint8
		if label is not None and binary_mask.sum().item() == 0:
			binary_mask = (label >= 0.5)

		if self.TEST is not None:
			self.left = self.margin
			self.right = self.margin
			self.top = self.margin
			self.bottom = self.margin
		else:
			self.update_margin()
			
		if binary_mask.sum().item() == 0: # avoid this by pre-condition in TEST 'F'
			minA = 0
			maxA = W
			minB = 0
			maxB = H
			self.no_forward = True
		else:
			if N > 1:
				mask = torch.zeros(size = (N, C, W, H))
				for n in range(N):
					cur_mask = binary_mask[n, :, :, :]
					arr = torch.nonzero(cur_mask)
					minA = arr[:, 1].min().item()
					maxA = arr[:, 1].max().item()
					minB = arr[:, 2].min().item()
					maxB = arr[:, 2].max().item()
					bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
			int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
					mask[n, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
				saliency_data = saliency_data * mask.cuda()

			arr = torch.nonzero(binary_mask)
			minA = arr[:, 2].min().item()
			maxA = arr[:, 2].max().item()
			minB = arr[:, 3].min().item()
			maxB = arr[:, 3].max().item()
			self.no_forward = False

		bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
			int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
		cropped_image = saliency_data[:, :, bbox[0]: bbox[1], \
			bbox[2]: bbox[3]]
		
		if self.no_forward == True and self.TEST == 'F':
			cropped_image = torch.zeros_like(cropped_image).cuda()

		crop_info = np.zeros((1, 4), dtype = np.int16)
		crop_info[0] = bbox
		crop_info = torch.from_numpy(crop_info).cuda()

		return cropped_image, crop_info

	def update_margin(self):
		MAX_INT = 256
		if random.randint(0, MAX_INT - 1) >= MAX_INT * self.prob:
			self.left = self.margin
			self.right = self.margin
			self.top = self.margin
			self.bottom = self.margin
		else:
			a = np.zeros(self.batch * 4, dtype = np.uint8)
			for i in range(self.batch * 4):
				a[i] = random.randint(0, self.margin * 2)
			self.left = int(a[0: self.batch].sum() / self.batch)
			self.right = int(a[self.batch: self.batch * 2].sum() / self.batch)
			self.top = int(a[self.batch * 2: self.batch * 3].sum() / self.batch)
			self.bottom = int(a[self.batch * 3: self.batch * 4].sum() / self.batch)

	def uncrop(self, crop_info, cropped_image, image):
		uncropped_image = torch.ones_like(image).cuda()
		uncropped_image *= (-9999999)
		bbox = crop_info[0]
		uncropped_image[:, :, bbox[0].item(): bbox[1].item(), bbox[2].item(): bbox[3].item()] = cropped_image
		return uncropped_image

def get_parameters(model, coarse=True, bias=False, parallel=False):
	print('coarse, bias', coarse, bias)
	if parallel:
		for name, mod in model.named_children():
			print('parallel', name)
			model = mod
			break
	for name, mod in model.named_children():
		if name == 'coarse_model' and coarse \
			or name in ['saliency1', 'saliency2', 'fine_model'] and not coarse:
			# or name in ['saliency1', 'saliency2', 'fine_model', 'fine_model_ema'] and not coarse:
			print(name)
			for n, m in mod.named_modules():
				if isinstance(m, nn.Conv2d):
					print(n, m)
					if bias and m.bias is not None:
						yield m.bias
					elif not bias:
						yield m.weight
				elif isinstance(m, nn.ConvTranspose2d):
					# weight is frozen because it is just a bilinear upsampling
					if bias:
						assert m.bias is None

class DSC_loss(nn.Module):
	def __init__(self):
		super(DSC_loss, self).__init__()
		self.epsilon = 0.000001
		return
	def forward(self, pred, target): # soft mode. per item. 
		batch_num = pred.shape[0]
		pred = pred.contiguous().view(batch_num, -1)
		target = target.contiguous().view(batch_num, -1)
		DSC = (2 * (pred * target).sum(1) + self.epsilon) / \
				   ((pred + target).sum(1) + self.epsilon)
		return 1 - DSC.sum() / float(batch_num)
