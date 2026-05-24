import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# 指定字体（根据系统任选其一）
# Windows 黑体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']        # 或 'Microsoft YaHei'
# macOS / Linux 可用 'PingFang SC'、'Noto Sans CJK SC' 等

# 解决负号显示为方块
mpl.rcParams['axes.unicode_minus'] = False

# --------------------------------
# 1. 准备数据与标签
# --------------------------------
import pandas as pd
row_labels = ['cereal',
'fmt',
'jemalloc',
'json_for_modern_cpp',
'jsoncpp',
'libpng',
'libxml2',
'libzmq',
'lodepng',
'lz4',
'mimalloc',
'opentelemetry',
'pugixml',
'tinyxml2',
'zlib',
'zstd',
'Average']
# ,'METEOR','BLEU-4','ROUGE_L_F','ROUGE_L_P','ROUGE_L_R'
col_labels = ['LLM_as_judge', 'Semantic Similar','METEOR','BLEU-4','ROUGE_L']
dim_labels = ['Func','Module','Repo']
colors = ['#4c72b0', '#55a868', '#c44e52']

# 读取Excel文件
data = []
for name in col_labels:
    # 读取某个指标name的数据
    temp = pd.read_excel('diff-repo.xlsx',sheet_name=name)
    temp_metric = []
    for i in range(len(temp)):
        temp_dim = [round(temp.iloc[i,1],2),round(temp.iloc[i,2],2),round(temp.iloc[i,3],2)]
        temp_metric.append(copy.deepcopy(temp_dim))

    data.append(copy.deepcopy(temp_metric))

data = np.array(data, dtype=np.float64)
data = data.transpose(1, 0, 2)

rows, cols = 17, 5
# np.random.seed(42)
# data = np.random.randint(10, 100, size=(rows, cols, 3))   # (类别, 指标, 维度)



# --------------------------------
# 2. 创建主网格 + 标签区域
# --------------------------------
fig = plt.figure(figsize=(cols*4 + 1.8, rows*0.9 + 0.8))

# 主网格
gs_main = fig.add_gridspec(nrows=rows, ncols=cols,
                           left=0.15, right=0.88,
                           bottom=0.05, top=0.95,
                           wspace=0.08, hspace=0.08)

# 行标签（横向）
gs_row = fig.add_gridspec(nrows=rows, ncols=1,
                          left=0.02, right=0.14,
                          bottom=0.05, top=0.95,
                          wspace=0, hspace=0.08)
for r in range(rows):
    ax_row = fig.add_subplot(gs_row[r, 0])
    ax_row.text(0.5, 0.5, row_labels[r],
                ha='center', va='center',
                fontsize=15, rotation=0)
    ax_row.set_xticks([]); ax_row.set_yticks([])
    ax_row.spines[:].set_visible(False)

# 列标签
gs_col = fig.add_gridspec(nrows=1, ncols=cols,
                          left=0.15, right=0.88,
                          bottom=0.96, top=0.99,
                          wspace=0.08, hspace=0)
for c in range(cols):
    ax_col = fig.add_subplot(gs_col[0, c])
    ax_col.text(0.5, 0.5, col_labels[c],
                ha='center', va='center', fontsize=15)
    ax_col.set_xticks([]); ax_col.set_yticks([])
    ax_col.spines[:].set_visible(False)

# 子图
axes = [[fig.add_subplot(gs_main[r, c]) for c in range(cols)]
        for r in range(rows)]
for r in range(rows):
    for c in range(cols):
        ax = axes[r][c]
        values = data[r, c]
        bars = ax.barh(dim_labels, values, color=colors, height=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', ha='left', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=7)

# 统一图例
handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors]
# fig.legend(handles, dim_labels,
#            loc='center right',
#            bbox_to_anchor=(0.99, 0.5),
#            frameon=False, fontsize=10)
fig.legend(handles, dim_labels,
           loc='lower center',           # 正下方
           bbox_to_anchor=(0.5, -0.02),  # 稍微往下一点
           ncol=3,                       # 三列并排
           frameon=False,
           fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('diff-repo-grand.png', dpi=300, bbox_inches='tight')
plt.show()