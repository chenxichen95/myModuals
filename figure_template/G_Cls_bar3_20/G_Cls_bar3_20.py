import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 载入数据
    np.random.seed(0)
    data = np.random.random([6, 5])*0.3 + 0.3

    # 参数设置
    xlabel = r'$\sigma$'
    xticks = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$']
    ylabel = r'$\mu$'
    yticks = [r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$']
    zlabel = 'NMI'
    title = 'G_Cls_bar3_20+参数搜索'

    # 开始画图
    fig = plt.figure(figsize=(20, 12), dpi=100)
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
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
    _xx, _yy = np.meshgrid(_x, _y)
    bottom = np.zeros_like(_xx)
    width = depth = 0.5
    for i in range(data.shape[0]):
        ax.bar3d(_xx[:, i], _yy[:, i], bottom[:, i], width, depth, data[i, :], shade=True, edgecolor='#000000')
    ax.set_title(title, titleFont)

    # 坐标轴设置
    ax.set_ylabel(f'\n\n{ylabel}', **LabelFont, horizontalalignment='right', verticalalignment='center')
    ax.set_xlabel(f'\n\n{xlabel}', **LabelFont)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f'{zlabel}\n', **LabelFont, rotation=90, horizontalalignment='right', verticalalignment='center')

    ax.invert_yaxis()

    plt.xticks(_x, xticks)
    plt.yticks(_y, yticks)
    ax.set_xticklabels(xticks, tickFont)
    ax.set_yticklabels(yticks, tickFont)

    ax.tick_params(labelsize=tickFont['fontsize'])

    # 调试摄像头角度
    # fig.show()
    # print('ax.azim {}'.format(ax.azim))
    # print('ax.elev {}'.format(ax.elev))
    # print('end')
    elev = 29.94285116604152
    azim = 40.04114325732064
    ax.view_init(elev=elev, azim=azim)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()


