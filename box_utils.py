import torch
import math
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageDraw
import sys

class BoxUtils:
    def __init__(self, num_cells, w, h, dtype=torch.FloatTensor, autograd_is_on=False):
        self.num_cells = num_cells
        self.w = w
        self.h = h
        self.grid_size = self.w/ self.num_cells
        self.dtype = dtype
        self.autograd_is_on = autograd_is_on
        self.preprocess = transforms.Compose([
            transforms.Scale((self.w, self.h)),
            transforms.ToTensor()
            ])
        self.swap_op = torch.Tensor([[0,1],[-1,0]]).type(dtype)
        self.corners = torch.Tensor([[-1,1],[1, 1],[1,-1],[-1,-1]]).type(self.dtype)

        if self.autograd_is_on:
            self.swap_op = Variable(self.swap_op)
            self.corners = Variable(self.corners)


    def get_oriented_box(self, xy_corners):
        # xy_corners is 8 x 2.
        # from 8 (x,y) xy_corners clockwise, get x, y, w, h, cos theta, sin theta
        # first point is top left
        center = xy_corners.sum(0)/4
        w = (xy_corners[0] - xy_corners[1]).pow(2).sum(0).sqrt()
        h = (xy_corners[1] - xy_corners[2]).pow(2).sum(0).sqrt()
        
        cos_sin = ((xy_corners[2] - xy_corners[1]) / w[0]).view(1, 2)

        return torch.cat((center, w, h, cos_sin), 1).squeeze()

    def get_oriented_boxes(self, xy_boxes):
        oobs = torch.zeros(xy_boxes.size(0), 6)
        for i in range(oobs.size(0)):
            oobs[i] = self.get_oriented_box(xy_boxes[i])

        return oobs

    def normalize_boxes(self, boxes, indices):
        # boxes is numboxes x 6
        # input boxes is (x ,y ,w, h, sin t, cos t). map to (x -u ) / delta, 
        # (y - v) / delta, w/W, h/H, cos t, sin t
        # indices is index into the grid cell of the image.
        nboxes = torch.zeros(boxes.size())
        for i in range(boxes.size(0)):
            nboxes[i] = self.normalize_box(boxes[i], indices[i])
        
        return nboxes

    def normalize_box(self, box, index):
        ones = torch.ones(1,2).type(self.dtype)
        ones = Variable(ones) if self.autograd_is_on else ones

        lower_corner = self.get_lower_corner(index[0], index[1])
        box[:2] = (box[:2] - lower_corner).div(self.grid_size * ones)
        box[2] = box[2]/self.w 
        box[3] = box[3]/self.h

        return box

    def get_lower_corner(self, xindex, yindex):
        x = math.floor(self.grid_size * xindex)
        y = math.floor(self.grid_size * yindex)

        t = torch.Tensor([x, y]).type(self.dtype)

        if self.autograd_is_on:
            t = Variable(t)
        
        return t

    def get_obb_pts(self, obb):
        # get xy corners of axis aligned box and rotate
        # obb is numboxes x 6
        boxcenter = obb[:,:2]
        wh = obb[:,2:4]

        # Get four corners of axis aligned box
        #corners = torch.Tensor([[-1,1],[1, 1],[1,-1],[-1,-1]]).type(self.dtype)
        corners = self.corners.unsqueeze(0)

        corners = corners.expand(boxcenter.size(0),4,2)
        corners = corners * wh.unsqueeze(1).expand_as(corners)/2
        corners = corners + boxcenter.unsqueeze(1).expand_as(corners)

        # Now rotate each corner by theta
        cos_sin = obb[:,4:6]
        negsin_cos = torch.mm(obb[:,4:6], self.swap_op)
        # construct rotation matrices
        R = torch.cat((cos_sin.unsqueeze(1), negsin_cos.unsqueeze(1)),1)

        pts = torch.bmm(corners, R)

        return pts

    def get_axis_bb(self, oobs):
        # input obb numboxes x [x, y, w, h, cos t, sin t]
        # output wh. wh is batchSize x num_cells x num_cells x 2
        # oobs has sqrt(x), sqrt(y), sqrt(w), sqrt(h), cost, sint

        obb = oobs.clone()
        pts = self.get_obb_pts(obb)
        min_values, _ = pts.min(1)
        max_values, _ = pts.max(1)

        min_x = min_values[:,:,0].transpose(0,1)
        max_x = max_values[:,:,0].transpose(0,1)
        min_y = min_values[:,:,1].transpose(0,1)
        max_y = max_values[:,:,1].transpose(0,1)
        
        return torch.cat((min_x, min_y, max_x, max_y), 1)

    def iou(self, boxes1, boxes2):
        # computes intersection over union of two axis aligned boxes
        # Input is number_of_boxes x 4. boxes1 and boxes2 are same in number.
        # each box is (xmin, ymin, xmax, ymax) where x, y is left lower corner
        # output is numboxes x 1. The ith element in this output is iou between
        # ith box in boxes1 and ith box in boxes2

        if boxes1.size(0) == boxes2.size(0):
            sys.exit("boxes size should be same")
        # check intersection area
        size = boxes1[:,0].size()

        x2 = torch.max(boxes1[:,2], boxes2[:,2])
        x1 = torch.min(boxes1[:,0], boxes2[:,0])
        y2 = torch.max(boxes1[:,3], boxes2[:,3])
        y1 = torch.min(boxes1[:,1], boxes2[:,1])

        inter = (x2 - x1 ) * (y2- y1 )
        boxes1_area = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        boxes2_area = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

        iou = inter/(boxes1_area + boxes2_area - inter)
        iou[(x2 -x1 <= 0)] = 0
        iou[(y2 - y1 <= 0)] = 0

        return iou

    def get_cell_indices(self, point):
        # from a point get the cell indices it belongs to

        return [max(0, min(self.num_cells-1, math.floor(float(point[0]) /self.grid_size))), max(0, min(self.num_cells-1, math.floor(float(point[1])/self.grid_size)))]

    def get_boxes_from_pred(self, pred, cutoff):
        # pred is batchSize x num_cells x num_cells x 7. Coords is c, x, y, w, h
        # get the cells for which the confidence is higher than cut off
        boxes = []

        for i in range(pred.size(0)):
            boxes_for_image = []
            boxes.append(boxes_for_image)

            mask = pred[i, :, :, 0]**2 > cutoff
            non_zero_indices = zip(*mask.cpu().numpy().nonzero())

            for j, k in non_zero_indices:
                lower_corner = self.get_lower_corner(j, k)
                c = pred[i,j,k,0]
                x = lower_corner[0] + pred[i,j,k,1]**2 * self.grid_size
                y = lower_corner[1] + pred[i,j,k,2]**2 * self.grid_size
                w = self.w * pred[i,j,k,3]**2
                h = self.h * pred[i,j,k,4]**2
                cost = pred[i,j,k,5]
                sint = pred[i,j,k,6]
                if type(c) == type(torch.cuda.FloatTensor):
                    boxes_for_image.append(torch.Tensor([c.data[0],x.data[0],y.data[0],w.data[0],h.data[0],cost.data[0],sint.data[0]]).type(self.dtype))
                else:
                    boxes_for_image.append(torch.Tensor([c,x,y,w,h,cost,sint]).type(self.dtype))

        return boxes

