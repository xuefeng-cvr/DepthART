import torch
import numpy as np
import cv2


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


# RunningAverage 类 负责维护一个数值的实时平均值。每次调用 append 方法时，它会更新这个平均值，并返回当前的平均值。
class RunningAverage:
	def __init__(self):
		self.avg = 0
		self.count = 0

	def append(self, value):
		self.avg = (value + self.count * self.avg) / (self.count + 1)
		self.count += 1

	def get_value(self):
		return self.avg

class RunningAverageDict:
	def __init__(self):
		self._dict = None

	def update(self, new_dict):
		if self._dict is None:
			self._dict = dict()
			for key, value in new_dict.items():
				self._dict[key] = RunningAverage()

		for key, value in new_dict.items():
			self._dict[key].append(value)

	def get_value(self):
		return {key: value.get_value() for key, value in self._dict.items()}

# def compute_errors(gt, pred):
# 	assert pred.shape == gt.shape
	
# 	thresh = np.maximum((gt / pred), (pred / gt))
# 	d1 = (thresh < 1.25).mean()
# 	d2 = (thresh < 1.25 ** 2).mean()
# 	d3 = (thresh < 1.25 ** 3).mean()

# 	abs_rel = np.mean(np.abs(gt - pred) / gt)
# 	sq_rel = np.mean(((gt - pred) ** 2) / gt)

# 	rmse = (gt - pred) ** 2
# 	rmse = np.sqrt(rmse.mean())

# 	rmse_log = (np.log(gt) - np.log(pred)) ** 2
# 	rmse_log = np.sqrt(rmse_log.mean())

# 	err = np.log(pred) - np.log(gt)
# 	silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

# 	log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
	

# 	return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, rmse=rmse, log10=log10, rmse_log=rmse_log,
# 				silog=silog, sq_rel=sq_rel)


def compute_errors_and_save_errors(gt, pred,save_errors_path):
    assert pred.shape == gt.shape
    
    # 添加一个小的常数以避免对零值取对数
    epsilon = 1e-8
    gt = gt + epsilon
    pred = pred + epsilon
    
    # 过滤掉无效值
    valid_mask = (gt > epsilon) & (pred > epsilon)
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    
     # 创建一个白色图像
    vis_image = np.ones_like(gt) * 255
    
    # 将不在 d1 范围内的像素点标记为黑色
    vis_image[thresh >= 1.25] = 0
	# 将vis_image reshape为二维矩阵 224, 1184
    vis_image = vis_image.reshape((224, 1184))
    
    vis_image = vis_image.astype(np.uint8)
    cv2.imwrite(save_errors_path, vis_image)

    return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, rmse=rmse, log10=log10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def compute_errors(gt, pred):
    assert pred.shape == gt.shape
    
    # 添加一个小的常数以避免对零值取对数
    epsilon = 1e-8
    gt = gt + epsilon
    pred = pred + epsilon
    
    # 过滤掉无效值
    valid_mask = (gt > epsilon) & (pred > epsilon)
    gt = gt[valid_mask]
    pred = pred[valid_mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    
    return dict(d1=d1, d2=d2, d3=d3, abs_rel=abs_rel, rmse=rmse, log10=log10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


	
def recover_metric_depth(pred, depth, valid_mask):
    # recover_metric_depth(pred,depth,valid_mask)
    gt_mask = depth[valid_mask] 
    # 对gt_mask取反(先将gt转换到视差空间下)
    gt_mask = 1.0 / gt_mask
    pred_mask = pred[valid_mask]

    # 标准化数据
    pred_mask_mean = np.mean(pred_mask)
    pred_mask_std = np.std(pred_mask)
    gt_mask_mean = np.mean(gt_mask)
    gt_mask_std = np.std(gt_mask)

    pred_mask = (pred_mask - pred_mask_mean) / pred_mask_std
    gt_mask = (gt_mask - gt_mask_mean) / gt_mask_std

    a, b = np.polyfit(pred_mask, gt_mask, deg=1)

    # 反标准化
    pred = (pred - pred_mask_mean) / pred_mask_std
    pred = a * pred + b
    pred = pred * gt_mask_std + gt_mask_mean

    # 再转换到深度空间下
    pred = 1.0 / pred 

    return pred





def compute_scale_and_shift(prediction, target, mask):
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction) # , axis=(1, 2)
    a_01 = np.sum(mask * prediction) # , axis=(1, 2)
    a_11 = np.sum(mask) # , axis=(1, 2)

    # Right-hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target) # , axis=(1, 2)
    b_1 = np.sum(mask * target) # , axis=(1, 2)

    # Solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def align_depth_least_square(prediction, target, mask, depth_cap=10):

    # Transform predicted disparity to aligned depth
    target_disparity = np.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]
   
    scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
    prediction_aligned = scale * prediction + shift
    # 限制最大深度为10m
    disparity_cap = 1.0 / depth_cap
    prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap 

    prediction_depth = 1.0 / prediction_aligned

    return prediction_depth
def submap_align(submap2,submap1):
    target = submap1[:, 192:224]
    prediction = submap2[:, 0:32]

    mask = (target > 0)
    scale, shift = compute_scale_and_shift(prediction, target, mask)
    prediction_aligned = scale * submap2 + shift

    return prediction_aligned

# 拼接子图
def blend_overlap(full_pred, pred1, pred2, pred3, pred4, pred5, pred6):
    """
    将子图拼接到全图中，并处理重叠区域的加权平均。
    :param full_pred: 全图张量
    :param pred1: 子图1预测结果
    :param pred2: 子图2预测结果
    :param pred3: 子图3预测结果
    :param pred4: 子图4预测结果
    :param pred5: 子图5预测结果
    :param pred6: 子图6预测结果
    :param overlap: 重叠区域大小
    """
    full_pred[ :, 0:224] = pred1
    full_pred[ :, 192:416] = pred2
    full_pred[ :, 384:608] = pred3
    full_pred[ :, 576:800] = pred4
    full_pred[ :, 768:992] = pred5
    full_pred[ :, 960:1184] = pred6

    full_pred[ :, 192:224] =  (pred1[ :, 192:224] + pred2[ :, :32] ) / 2
    full_pred[ :, 384:416] =  (pred2[ :, 192:224] + pred3[ :, :32] ) / 2
    full_pred[ :, 576:608] =  (pred3[ :, 192:224] + pred4[ :, :32] ) / 2
    full_pred[ :, 768:800] =  (pred4[ :, 192:224] + pred5[ :, :32] ) / 2
    full_pred[ :, 960:992] =  (pred5[ :, 192:224] + pred6[ :, :32] ) / 2

    return full_pred

