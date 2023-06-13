from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, five_paths_from_folder
from basicsr.data.transforms import augment, five_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CGCImageDataset(data.Dataset):
    def __init__(self, opt):
        super(CGCImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.dis_folder, self.seg_folder = opt['dataroot_dis'], opt['dataroot_seg']
        self.sobel_folder = opt['dataroot_sobel']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.paths = five_paths_from_folder([self.lq_folder, self.gt_folder, self.seg_folder, self.dis_folder, self.sobel_folder],
                                            ['lq', 'gt', 'seg', 'dis', 'sobel'], self.filename_tmpl)


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        #self.paths = [4000, 4]
        # print(self.paths[1200])
        # print('----------')
        # print(type(self.paths[1200]))
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        seg_path = self.paths[index]['seg_path']
        img_bytes =self.file_client.get(seg_path, 'seg')
        img_seg = imfrombytes(img_bytes, float32=True)

        dis_path = self.paths[index]['dis_path']
        img_bytes = self.file_client.get(dis_path, 'dis')
        img_dis = imfrombytes(img_bytes, float32=True)

        sobel_path = self.paths[index]['sobel_path']
        img_bytes = self.file_client.get(sobel_path, 'sobel')
        img_sobel = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size'] #尺寸 96 * 96
            # random crop
            img_gt,img_lq,img_seg,img_dis,img_sobel = five_random_crop(img_gt,img_lq,img_seg,img_dis,img_sobel,gt_size,scale,gt_path)
            # flip, rotation
            img_gt,img_lq,img_seg,img_dis,img_sobel = augment([img_gt,img_lq,img_seg,img_dis,img_sobel], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
            img_seg = rgb2ycbcr(img_seg, y_only=True)[..., None]
            img_dis = rgb2ycbcr(img_dis, y_only=True)[..., None]
            img_sobel = rgb2ycbcr(img_sobel, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq ,img_seg, img_dis, img_sobel = img2tensor([img_gt, img_lq, img_seg, img_dis, img_sobel], bgr2rgb=True, float32=True)
        #print(type(img_gt)) #tensor
        img_dis = img_dis[0].unsqueeze(0)
        img_sobel = img_sobel[0].unsqueeze(0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_seg, self.mean, self.std, inplace=True)
            normalize(img_dis, self.mean, self.std, inplace=True)
            normalize(img_sobel, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'seg':img_seg, 'dis':img_dis, 'sobel':img_sobel,
                'lq_path': lq_path, 'gt_path': gt_path, 'seg_path':seg_path, 'dis_path':dis_path, 'sobel_path':sobel_path}

    def __len__(self):
        return len(self.paths)