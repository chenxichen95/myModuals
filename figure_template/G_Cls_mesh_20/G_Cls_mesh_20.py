import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np

if __name__ == "__main__":
    # 载入数据
    data = scio.loadmat('./DATA.mat')['DATA']

    # 参数设置
    xlabel = r'$\lambda$'
    xticks = [str(i) for i in range(2, 22, 2)]
    ylabel = r'$\mu$'
    yticks = ['0.01', '0.1', '0.5', '1']
    zlabel = 'NMI'
    title = 'G_Cls_mesh_20+参数搜索'


    # 开始画图
    fig = plt.figure(figsize=(20, 12), dpi=100)
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

    _x = np.arange(data.shape[1])
    _y = np.arange(data.shape[0])
    _xx, _yy = np.meshgrid(_x, _y)

    ax.set_title(title, titleFont)

    ax.set_zlim(0.3, 0.6)

    ax.plot_wireframe(_xx, _yy, data)

    # 坐标轴设置
    ax.set_ylabel(f'\n\n{ylabel}', **LabelFont)
    ax.set_xlabel(f'\n\n{xlabel}', **LabelFont)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f'{zlabel}\n\n\n', **LabelFont, rotation=90)

    plt.xticks([int(i) for i in xticks], xticks)
    plt.yticks(_y, xticks)
    ax.set_zticklabels([f'{i:.2f}    ' for i in ax.get_zticks()], tickFont)
    ax.set_xticklabels(xticks, tickFont)
    ax.set_yticklabels(yticks, tickFont)

    ax.invert_xaxis()
    ax.tick_params(axis='z', labelsize=tickFont['fontsize'])

    # 调试摄像头角度
    # fig.show()
    # print('ax.azim {}'.format(ax.azim))
    # print('ax.elev {}'.format(ax.elev))
    # print('end')
    elev = 27.694044549778255
    azim = 63.066610808546216
    ax.view_init(elev=elev, azim=azim)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()


