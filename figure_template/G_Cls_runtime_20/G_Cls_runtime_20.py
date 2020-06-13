import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p']

if __name__ == '__main__':
    # 载入数据
    data = np.array(
        [
            [ 0.0110, 0.0012, 0.0011, 0.0013, 0.0013],  # NMF   11个方法
            [0.0153, 0.0016, 0.0015, 0.0017, 0.0018],   # GNMF
            [0.0237, 0.0020, 0.0020, 0.0022, 0.0023],   #  FeatConcate
            [0.0673, 0.0063, 0.0072, 0.0091, 0.0083],   # ColNMF
            [0.0542, 0.0059, 0.0049, 0.0060, 0.0075],   # MultiNMF
            [0.7971, 0.7580, 0.7543, 0.7382, 0.7675],   # RMKMC
            [0.0489, 0.0291, 0.0249, 0.0265, 0.0293],    # CoRegSPC
            [0.1767, 0.0198, 0.0187, 0.0280, 0.0346],    # KMLRSSC
            [0.0229, 0.0021, 0.0022, 0.0020, 0.0023],    # DiNMF
            [0.0235, 0.0022, 0.0024, 0.0026, 0.0026],   # LP-DiNMF
            [0.0260, 0.0024, 0.0026, 0.0029, 0.0028],    # NMF-CC
        ]
    )

    # 参数设置
    legends = ['NMF', 'GNMF', 'FNMF', 'ColNMF', 'MultiNMF', 'RMKMC', 'CoRegSPC', 'KMLRSSC', 'DiNMF', 'LP-DiNMF', 'NMF-CC']
    xticks = ['ORL', 'Cornell', 'Texas', 'Washington', 'Wisconsin']
    yticks = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
    ylabel = 'Run Time(s)'
    title = 'G_Cls_runtime_20+结果对比'
    log_y_ticks = [0.001, 0.01, 0.1, 1]

    # 绘图
    fig = plt.figure(figsize=(12, 12), dpi=100)
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
        'size': 30,
    }

    for index in range(data.shape[0]):
        ax.plot(np.arange(data.shape[1]), data[index, :], linewidth=4, marker=markers[index], markersize=18)

    # ax.set_yticklabels(ax.get_yticks(), **tickFont)
    plt.xticks(np.arange(data.shape[1]), xticks)
    ax.set_xticklabels(xticks, LabelFont)

    ax.set_yscale('log')
    plt.yticks([0.001, 0.01, 0.1, 1], yticks)
    ax.set_yticklabels(yticks, **tickFont)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(which='both', direction='in', width=2)
    ax.set_ylabel(f'{ylabel}', **LabelFont)

    ax.legend(legends,  framealpha=1, prop=legendFont, loc=[1, 0])

    ax.set_title(title, titleFont)

    # 保存图片
    fig.savefig('./figure.jpg', format='jpg', bbox_inches='tight')

    # 显示图片
    plt.show()
