import torch
import matplotlib.pyplot as plt
x = torch.tensor([0, 1, 2, 3, 4])
y = torch.tensor([0, 1, 2, 3, 4])


grid_x, grid_y = torch.meshgrid(x, y)

# a = torch.arange(9).reshape((1,1,3,3)).float()
a = torch.ones(25).reshape((1,1,5,5)).float().requires_grad_(True)
# plt.imshow(a[0][0])
# plt.show()
gt = torch.ones(25).reshape((1,1,5,5)).float()*2
pix_coords = torch.stack([grid_x,grid_y],dim=-1).float()
print(pix_coords.shape)
pix_coords+=2
pix_coords = pix_coords.view(1, 5, 5, 2)
# pix_coords = pix_coords.permute(0, 2, 3, 1)
pix_coords[..., 0] /= (5 - 1)
pix_coords[..., 1] /= (5 - 1)
pix_coords = (pix_coords - 0.5) * 2
# b = torch.nn.functional.grid_sample(a,pix_coords,mode='bilinear',padding_mode='zeros',align_corners=True)
b = torch.nn.functional.grid_sample(a,pix_coords,mode='bilinear',padding_mode='border',align_corners=True)
mask = torch.nn.functional.grid_sample(a,pix_coords,mode='bilinear',padding_mode='zeros',align_corners=True)
mask = torch.ones_like(a)
c=b+1

# print(b[0][0])
d = (mask*(c-gt)).sum()
d.backward()
print(b.grad)
print(a.grad)
plt.imshow(b.detach().numpy()[0][0])
plt.colorbar()
plt.show()

exit()
