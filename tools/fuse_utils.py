# scripts/fuse_utils.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def clip_xy(pts, W, H):
    out = pts.copy()
    out[:,0] = np.clip(out[:,0], 0, W-1)
    out[:,1] = np.clip(out[:,1], 0, H-1)
    return out

def bilinear_sample(img, x, y):
    x = float(np.clip(x, 0, img.shape[1]-1))
    y = float(np.clip(y, 0, img.shape[0]-1))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, img.shape[1]-1), min(y0+1, img.shape[0]-1)
    dx, dy = x - x0, y - y0
    v00 = img[y0,x0]; v10 = img[y0,x1]; v01 = img[y1,x0]; v11 = img[y1,x1]
    return (v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy)

def geometric_median(X, eps=1e-6, max_iter=200):
    y = X.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        w = 1.0 / np.clip(d, eps, None)
        y_new = (w[:,None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps: break
        y = y_new
    return y

def backproject_points(px, py, Z_rel, fx, fy, cx, cy):
    X = (px - cx) * Z_rel / fx
    Y = (py - cy) * Z_rel / fy
    return np.stack([X, Y, Z_rel], axis=-1)

def project_point(P, fx, fy, cx, cy):
    X, Y, Zp = P
    Zp = float(max(Zp, 1e-6))
    u = cx + fx * (X / Zp)
    v = cy + fy * (Y / Zp)
    return np.array([u, v], dtype=np.float32)

def plot_overlay_depth_centers(Z, pts2d, centers2d, centers3d_uv, save_path):
    H, W = Z.shape
    plt.figure(figsize=(8,6))
    plt.imshow(np.ma.masked_invalid(Z), cmap='plasma')
    plt.scatter(pts2d[:,0], pts2d[:,1], s=28, c='white', edgecolor='k', linewidth=0.3, label='keypoints')
    (c2d, g2d) = centers2d
    (c3uv, g3uv) = centers3d_uv
    plt.scatter([c2d[0]],[c2d[1]], s=120, marker='*', c='lime',   edgecolor='k', label='2D centroid')
    plt.scatter([g2d[0]],[g2d[1]], s=100, marker='X', c='cyan',   edgecolor='k', label='2D geom.med.')
    plt.scatter([c3uv[0]],[c3uv[1]], s=90, marker='o', facecolors='none', edgecolors='yellow',  linewidth=2.0, label='proj 3D centroid')
    plt.scatter([g3uv[0]],[g3uv[1]], s=90, marker='o', facecolors='none', edgecolors='magenta', linewidth=2.0, label='proj 3D geom.med.')
    plt.title("Depth + keypoints + 2D/3D centers")
    plt.legend(fontsize=8, markerscale=0.7, frameon=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def plot_hist_depth(kz, save_path):
    plt.figure(figsize=(6,4))
    plt.hist(kz, bins=min(len(kz), 12))
    plt.xlabel("Depth")
    plt.ylabel("count")
    plt.title("Keypoint-depth distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def plot_points3d(P3, centers3d, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P3[:,0], P3[:,1], P3[:,2], s=30)
    ax.scatter([centers3d[0][0]],[centers3d[0][1]],[centers3d[0][2]], s=100, marker='*', label='3D centroid')
    ax.scatter([centers3d[1][0]],[centers3d[1][1]],[centers3d[1][2]], s=90,  marker='X', label='3D geom. med.')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



