import numpy as np

dims = np.array([15.89997, 24.85395, 7.41958])

sy = dims[1]
sx = dims[0]
sz = dims[2]

sx = sx * 0.5
sy = sy * 0.5
sz = sz * 0.5

points_3D = np.array([[  79.49984741  ,124.2697525   ,-37.09790039],
 [  79.49984741 , 124.2697525   , 37.09790039],
 [ -79.49984741 , 124.2697525   , 37.09790039],
 [ -79.49984741 , 124.2697525   ,-37.09790039],
 [  79.49984741 ,-124.2697525   ,-37.09790039],
 [  79.49984741 ,-124.2697525    ,37.09790039],
 [ -79.49984741 ,-124.2697525    ,37.09790039],
 [ -79.49984741 ,-124.2697525   ,-37.09790039]] )/10

# forward = f * sy * 0.5
# up = u * sz * 0.5
# right = r * sx * 0.5
center =  np.array([18.61763246, 6.76022065, 1.25508838])/10
A = np.array([[sy, sz, sx],
              [sy, sz, -sx],
              [sy, -sz, -sx],
              [sy, -sz, sx],
              [-sy, sz, sx],
              [-sy, sz, -sx],
              [-sy, -sz, -sx],
              [-sy, -sz, sx]])
b = np.zeros((8,3))
for i, p in enumerate(points_3D):
    b[i] = p - center

print(A, b)
A_t = A.transpose()
A_new = np.matmul(A_t , A)
b_new = np.matmul(A_t , b)
print(A_new.shape, b_new.shape)
x = np.linalg.solve(A_new, b_new)
print(x)
forward = x[0]
up = x[1]
right = x[2]
vertices = np.array([center + forward + up + right,      # Front Top Right
center + forward + up - right,      # Front Top Left
center + forward - up - right,      # Front Bottom Left
center + forward - up + right,      # Front Bottom Right
center - forward + up  + right,     # Rear Top Right
center - forward + up - right,      # Rear Top Left
center - forward - up - right,      # Rear Bottom Left
center - forward - up + right]) 

print(vertices)
# center + f * sy +  u * sz + r * sx,      # Front Top Right
# center + f * sy +  u * sz - r * sx,      # Front Top Left
# center + f * sy -  u * sz - r * sx,      # Front Bottom Left
# center + f * sy -  u * sz + r * sx,      # Front Bottom Right
# center - f * sy +  u * sz  + r * sx,     # Rear Top Right
# center - f * sy +  u * sz - r * sx,      # Rear Top Left
# center - f * sy -  u * sz - r * sx,      # Rear Bottom Left
# center - f * sy -  u * sz + r * sx,     # Rear Bottom Right