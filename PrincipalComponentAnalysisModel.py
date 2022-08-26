from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    data = np.load(filename)
    data = data - np.mean(data, axis=0)
    return data

def get_covariance(dataset):
    n = len(dataset)
    S = (1/(n-1))*np.dot(np.transpose(dataset), dataset)
    return S

def get_eig(S, m):
    w, v = np.linalg.eigh(S)
    w = w.reshape(1,1024)

    size = (m, m)
    w_new = np.zeros(size)

    v_w = np.vstack((v,w))
    v_w_sorted = v_w[:, v_w[1024, :].argsort()]

    v_new = np.flip(v_w_sorted[0:1024,:], axis=1)
    np.fill_diagonal(w_new, np.flip(v_w_sorted[1024, :], axis=0))

    return w_new, v_new[:, 0:m]

def get_eig_prop(S, prop):
    w, v = np.linalg.eigh(S)
    sum_w = float(sum(w))
    w = w.reshape(1,1024)

    v_w = np.vstack((v,w))
    v_w_sorted = v_w[:, v_w[1024, :].argsort()]

    for i in range(0, len(v_w_sorted[0])):
      if float(v_w_sorted[1024][1023 - i] / sum_w) <= prop :
        break

    size = (i,i)
    w_new = np.zeros(size)

    v_new = np.flip(v_w_sorted[0:1024,:], axis=1)
    np.fill_diagonal(w_new, np.flip(v_w_sorted[1024, :], axis=0))

    return w_new, v_new[:, 0:i]

def project_image(image, U):
    xipro = np.zeros((1,1024))
    for i in range(len(U[0])):
      uit = np.transpose(U[:,i]).reshape(1,1024)
      xi = image.reshape(1024,1)
      ui = U[:,i].reshape(1,1024)
      xipro += uit@xi@ui
    return xipro

def display_image(orig, proj):
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols =2)
    f.tight_layout(pad = 4.5)
    ax1.set_title('Original')    
    ax1.set_xticks(np.arange(0, 35, 10))
    ax1.set_yticks(np.arange(0, 35, 5))
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
	    label.set_fontsize(8)
    color_bar_1 = ax1.imshow(orig.reshape(32,32).T, aspect='equal')
    cax_1 = f.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    clb = f.colorbar(color_bar_1, ax=ax1, cax=cax_1, ticks=[-25, 0, 25, 50, 75, 100, 125, 150], )    
    clb.ax.tick_params(labelsize=8)

    ax2.set_title('Projection')
    ax2.set_xticks(np.arange(0, 35, 10))
    ax2.set_yticks(np.arange(0, 35, 5))
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
	    label.set_fontsize(8)
    color_bar_2 = ax2.imshow(proj.reshape(32,32).T, aspect='equal')
    cax_2 = f.add_axes([ax2.get_position().x1+0.01,ax2.get_position().y0,0.02,ax2.get_position().height])
    f.colorbar(color_bar_2, ax=ax2, cax=cax_2)
    clb = f.colorbar(color_bar_2, ax=ax2, cax=cax_2)
    clb.ax.tick_params(labelsize=8)
    plt.show()
