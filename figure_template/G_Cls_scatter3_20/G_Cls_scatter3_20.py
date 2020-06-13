import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np


if __name__ == "__main__":
    # 载入数据
    data = scio.loadmat('./DATA.mat')['DATA']

    # 参数设置
    xlabel = r'$\mu$'
    xticks = ['1', '5', '10', '50', '100', '500']
    ylabel = r'$Y$'
    yticks = ['1', '5', '10', '50', '100', '500']
    zticks = ['10', '30', '50', '70']
    zlabel = r'$\sigma$'
    title = 'G_Cls_scatter3_20+参数搜索'


    # 开始画图
    fig = plt.figure(figsize=(10, 18), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    titleFont = {
        'family': 'SimHei',
        'weight': 'bold',
        'fontsize': 50,
    }
    LabelFont = {
        'family': 'SimHei',
        'weight': 'bold',
        'fontsize': 30,
    }
    tickFont = {
        'family': 'SimHei',
        'weight': 'bold',
        'fontsize': 30,
    }


    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _z = np.arange(data.shape[2])
    _xx, _yy, _zz = np.meshgrid(_x, _y, _z)

    p = ax.scatter(_xx, _yy, _zz, c=data, s=150, marker='o', cmap='autumn')

    plt.xticks(_x, xticks)
    plt.yticks(_y, yticks)
    ax.set_zticks(_z)
    ax.set_xticklabels(xticks, tickFont)
    ax.set_yticklabels(yticks, tickFont)
    ax.set_zticklabels(zticks, tickFont)

    ax.tick_params(labelsize=tickFont['fontsize'])

    fig.colorbar(p, pad=0.01, shrink=0.7)

    ax.set_title(title, titleFont)

    # 坐标轴设置
    ax.set_ylabel(f'\n\n{ylabel}', **LabelFont)
    ax.set_xlabel(f'\n\n{xlabel}', **LabelFont)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f'{zlabel}\n', **LabelFont, rotation=90, horizontalalignment='right', verticalalignment='center')
    ax.set_zticklabels([f'{i}    ' for i in ax.get_zticks()], tickFont)
    ax.figure.axes[-1].set_yticklabels([f'{i}' for i in ax.figure.axes[-1].get_yticks()], tickFont)

    # 调试摄像头角度
    # fig.show()
    # print('ax.azim {}'.format(ax.azim))
    # print('ax.elev {}'.format(ax.elev))
    # print('end')
    elev = 14.146556279724962
    azim = -123.87096774193537
    ax.view_init(elev=elev, azim=azim)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()