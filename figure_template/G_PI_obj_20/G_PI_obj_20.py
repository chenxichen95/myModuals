import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.io as scio
matplotlib.rcParams['axes.unicode_minus']=False

if __name__ == '__main__':
    # 载入数据
    data = scio.loadmat('./DATA.mat')['obj']

    # 参数设置
    xlabel = 'Number of iteration #'
    ylabel = 'Objective function value'
    title = 'G_PI_obj_20+收敛'

    # 绘图
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_subplot(111)

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

    ax.plot(np.arange(data.shape[1]), data[0, :], linewidth=4)

    ax.set_xticklabels([int(i) for i in ax.get_xticks()], LabelFont)
    ax.set_yticklabels([int(i) for i in ax.get_yticks()], LabelFont)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(which='both', direction='in', width=2)

    ax.set_ylabel(f'{ylabel}', **LabelFont)
    ax.set_xlabel(f'{xlabel}', **LabelFont)

    ax.set_title(title, titleFont)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()