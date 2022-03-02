from operator import mod
import os
import sys
import time
from utils import *
from model import *
import ipdb
import pytorch_iou
import pytorch_gauss
from ramps import *
import torchvision.transforms as transforms


if __name__ == '__main__':
	data_path = sys.argv[1]
	current_fold = sys.argv[2]
	organ_number = int(sys.argv[3])
	low_range = int(sys.argv[4])
	high_range = int(sys.argv[5])
	slice_threshold = float(sys.argv[6])
	slice_thickness = int(sys.argv[7])
	organ_ID = int(sys.argv[8])
	plane = sys.argv[9]
	GPU_ID = int(sys.argv[10])
	learning_rate1 = float(sys.argv[11])
	learning_rate_m1 = int(sys.argv[12])
	learning_rate2 = float(sys.argv[13])
	learning_rate_m2 = int(sys.argv[14])
	crop_margin = int(sys.argv[15])
	crop_prob = float(sys.argv[16])
	crop_sample_batch = int(sys.argv[17])
	snapshot_path = os.path.join(snapshot_path, 'SIJ_training_' + \
		sys.argv[11] + 'x' + str(learning_rate_m1) + ',' + str(crop_margin))
	epoch = {}
	epoch['S'] = int(sys.argv[18])
	epoch['I'] = int(sys.argv[19]) 
	epoch['J'] = int(sys.argv[20])
	epoch['lr_decay'] = int(sys.argv[21])
	timestamp = sys.argv[22]

	if not os.path.exists(snapshot_path):
		os.makedirs(snapshot_path)
	
	Unet_weights = os.path.join(pretrained_model_path, 'unet_voc.pth')
	if not os.path.isfile(Unet_weights):
		raise RuntimeError('Please Download <http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU> from the Internet ...')

	from Data import DataLayer
	training_set = DataLayer(data_path=data_path, current_fold=int(current_fold), organ_number=organ_number, \
		low_range=low_range, high_range=high_range, slice_threshold=slice_threshold, slice_thickness=slice_thickness, \
		organ_ID=organ_ID, plane=plane)

	batch_size = 1
	os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_ID)
	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
	print(current_fold + plane, len(trainloader))
	print(epoch)

	RSTN_model = RSTN(crop_margin=crop_margin, \
		crop_prob=crop_prob, crop_sample_batch=crop_sample_batch)
	RSTN_snapshot = {}

	model_parameters = filter(lambda p: p.requires_grad, RSTN_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('model parameters:', params)

	#pdb.set_trace()
	for param in RSTN_model.coarse_model.parameters():
		param.detach_()

	optimizer = torch.optim.SGD(
		[
			{'params': get_parameters(RSTN_model, coarse=False, bias=False, parallel=False),
			'lr': learning_rate1 * 10},
			{'params': get_parameters(RSTN_model, coarse=False, bias=True, parallel=False),
			'lr': learning_rate1 * 20, 'weight_decay': 0}	
		],
		lr=learning_rate1,
		momentum=0.99,
		weight_decay=0.0005)

	criterion = DSC_loss()
	COARSE_WEIGHT = 1 / 3
	
	bce_loss = nn.BCELoss(size_average=True)
	gauss_loss = pytorch_gauss.Gauss(window_size=11,size_average=True)
	iou_loss = pytorch_iou.IOU(size_average=True)

	def update_ema_variables(model, ema_model, alpha):
		for ema_param, param in zip(ema_model.parameters(), model.parameters()):
			ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

	def update_variables(model, ema_model):
		for ema_param, param in zip(ema_model.parameters(), model.parameters()):
			ema_param.data = param.data

	def overall_loss(pred,target):		
		gauss_out = 1 - gauss_loss(pred, target)
		iou_out = iou_loss(pred, target)
		bce_out = bce_loss(pred, target)

		loss = bce_out + gauss_out + iou_out

		return loss

	RSTN_model = RSTN_model.cuda()
	RSTN_model.train()

	for mode in ['S','I','J']:
		if mode == 'S':
			RSTN_dict = RSTN_model.state_dict()
			pretrained_dict = torch.load(Unet_weights)

			w = pretrained_dict['final.weight'][20,:,:,:]
			b = pretrained_dict['final.bias'][20]
			pretrained_dict['final.weight'] = pretrained_dict['final.weight'][:3,:,:,:]
			pretrained_dict['final.bias'] = pretrained_dict['final.bias'][:3]
			pretrained_dict['final.weight'][0,:,:,:] = w
			pretrained_dict['final.bias'][0] = b
			pretrained_dict['final.weight'][1,:,:,:] = w
			pretrained_dict['final.bias'][1] = b
			pretrained_dict['final.weight'][2,:,:,:] = w
			pretrained_dict['final.bias'][2] = b
			# 1. filter out unnecessary keys
			pretrained_dict_coarse = {'coarse_model.' + k : v
					for k, v in pretrained_dict.items() 
					if 'coarse_model.' + k in RSTN_dict and 'score' not in k}
			pretrained_dict_fine = {'fine_model.' + k : v
					for k, v in pretrained_dict.items() 
					if 'fine_model.' + k in RSTN_dict and 'score' not in k}
			pretrained_dict_fine_ema = {'fine_model_ema.' + k : v
					for k, v in pretrained_dict.items() 
					if 'fine_model_ema.' + k in RSTN_dict and 'score' not in k}
			# 2. overwrite entries in the existing state dict
			RSTN_dict.update(pretrained_dict_coarse) 
			RSTN_dict.update(pretrained_dict_fine)
			RSTN_dict.update(pretrained_dict_fine_ema)

			# 3. load the new state dict
			RSTN_model.load_state_dict(RSTN_dict)
			print(plane + mode, 'load pre-trained unet_voc model successfully!')

		elif mode == 'I':
			print(plane + mode, 'load S model successfully!')
			# update_variables(RSTN_model.coarse_model, RSTN_model.fine_model_ema)
		elif mode == 'J':
			update_variables(RSTN_model.fine_model_ema, RSTN_model.fine_model)
			print(plane + mode, 'reload pretrained model for fine successfully!')

		else:
			raise ValueError("wrong value of mode, should be in ['S']")
		
		try:
			for e in range(epoch[mode]):
				total_loss = 0.0
				total_fine_loss = 0.0
				total_coarse_loss = 0.0
				start = time.time()
				for index, (image, label) in enumerate(trainloader):
					
					start_it = time.time()
					optimizer.zero_grad()
					image, label = image.cuda().float(), label.cuda().float()
					if mode == 'J':
						coarse_prob, fine_prob, fine_prob_gt = RSTN_model(image, e+1, label, mode=mode)
						fine_loss = overall_loss(fine_prob, label)
						fine_loss_gt = overall_loss(fine_prob_gt, label)
						loss = fine_loss + fine_loss_gt
					else:
						coarse_prob, fine_prob = RSTN_model(image, e+1, label, mode=mode)
						fine_loss = overall_loss(fine_prob, label)
						loss = fine_loss
					coarse_loss = overall_loss(coarse_prob, label)
					total_loss += loss.item()
					total_fine_loss += fine_loss.item()
					total_coarse_loss += coarse_loss.item()
					loss.backward()
					optimizer.step()

					if mode == 'S':
						update_ema_variables(RSTN_model.fine_model, RSTN_model.coarse_model, 0.999)
					elif mode == 'I':
						update_ema_variables(RSTN_model.fine_model, RSTN_model.coarse_model, 0.999)
					else:
						update_ema_variables(RSTN_model.fine_model, RSTN_model.fine_model_ema, 0.999)


					print(current_fold + plane + mode, "Epoch[%d/%d], Iter[%05d], Coarse/Fine Loss  %.4f/%.4f, Time Elapsed %.2fs" \
							%(e+1, epoch[mode], index, coarse_loss.item(),fine_loss.item(), time.time()-start_it))

					del image, label, fine_prob,coarse_prob, loss, fine_loss,coarse_loss

				print(current_fold + plane + mode, "Epoch[%d], Total Coarse/Fine Loss %.4f/%.4f, Time elapsed %.2fs" \
						%(e+1, total_coarse_loss / len(trainloader),total_fine_loss / len(trainloader),time.time()-start))
		except KeyboardInterrupt:
			print('!' * 10 , 'save before quitting ...')
		finally:
			if mode == 'J':
				update_variables(RSTN_model.fine_model_ema, RSTN_model.fine_model)
			snapshot_name = 'FD' + current_fold + ':' + \
				plane + mode + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
			RSTN_snapshot[mode] = os.path.join(snapshot_path, snapshot_name) + '.pkl'
			torch.save(RSTN_model.state_dict(), RSTN_snapshot[mode])
			print('#' * 10 , 'end of ' + current_fold + plane + mode + ' training stage!')