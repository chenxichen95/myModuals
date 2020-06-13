import matplotlib.pyplot as plt
import seaborn
import scipy.io as scio


if __name__ == "__main__":
    # 载入数据
    data = scio.loadmat('./DATA.mat')['DATA']

    # 参数设置
    xticklabels = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1', '5', '10', '50', '100', '500', '1000']
    yticklabels = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1', '5', '10', '50', '100', '500', '1000']
    ylabel = r'$\lambda_1$'
    xlabel = r'$\lambda_2$'
    title = 'G_Cls_imagesc_20+参数搜索 '

    # 开始画图
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

    seaborn.heatmap(data=data, vmin=None, vmax=None, cmap='winter', annot=None, fmt='.2g', annot_kws=None, cbar=True,  linewidths=False, linecolor=None, square=False, xticklabels=xticklabels, yticklabels=yticklabels, mask=None, ax=ax, cbar_kws=dict(pad=0.01))

    ax.set_ylabel(f'{ylabel}', **LabelFont, rotation='horizontal', horizontalalignment='left', verticalalignment='center')
    ax.set_xlabel(f'{xlabel}', **LabelFont)
    ax.set_xticklabels(xticklabels, tickFont, rotation=90, verticalalignment='top')
    ax.invert_yaxis()
    ax.set_yticklabels(yticklabels, tickFont, rotation='horizontal', verticalalignment='center')
    ax.figure.axes[-1].set_yticklabels([f'{i:.1f}' for i in ax.figure.axes[-1].get_yticks()], tickFont)

    ax.set_title(title, titleFont)

    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    ax.tick_params(direction='in', width=2)
    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()