import matplotlib.pyplot as plt
import numpy as np


# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

if __name__ == '__main__':
    # 载入数据
    np.random.seed(0)
    data = np.random.random([6, 5])*40 + 60

    # 参数设置
    legends = [r'$β=10^0$', r'$β=10^1$', r'$β=10^2$', r'$β=10^3$', r'$β=10^4$', r'$β=10^5$']
    xticks = ['case 1', 'case 2', 'case 3', 'case 4', 'case 5']
    ylabel = 'NMI(%)'
    width = 0.5
    title = 'G_Cls_bar_20+参数搜索'

    # 绘图
    fig = plt.figure(figsize=(20, 12), dpi=100)
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
    legendFont = {
        'family': 'SimHei',
        'weight': 'bold',
        'size': 20,
    }

    x = []
    for i in range(data.shape[1]):
        if i == 0:
            x.append(1)
        else:
            x.append(x[-1]+width*(data.shape[0]+1))

    x = np.array(x)
    _x = np.array(x)
    for index in range(data.shape[0]):
        _x = _x + width
        ax.bar(_x, data[index, :], width)

    # 设置 y 坐标轴的范围
    ax.set_ylim(30, 100)

    # 设置标题
    ax.set_title(title, titleFont)

    # 坐标轴设置
    plt.xticks([r + (width*data.shape[1]+1)/2 for r in x], xticks)
    ax.set_xticklabels(xticks, LabelFont)
    ax.set_ylabel(f'{ylabel}', **LabelFont)
    ax.set_yticklabels([i for i in ax.get_yticks()], **tickFont)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(direction='in', width=2)

    # 设置legend
    ax.legend(legends, loc='lower right', framealpha=1, prop=legendFont)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()

