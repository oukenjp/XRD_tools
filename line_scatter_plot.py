import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
#created by wang
# ================= 配置参数 =================
# 文件及其绘图信息列表，每个元素是字典
# color 和 linestyle 留空可自动分配
# use_yoffset 控制是否对该曲线做 y-offset
# use_normalize 控制是否对该曲线做归一化
files = [
    {"filename": "1.xy", "col_x": 0, "col_y": 1, "label": None, "type": "line", "color": None, "marker": None, "linestyle": None,},
    {"filename": "2.xy", "col_x": 0, "col_y": 1, "label": None, "type": "line", "color": None, "marker": None, "linestyle": None,},
    {"filename": "3.xy", "col_x": 0, "col_y": 1, "label": None, "type": "line", "color": None, "marker": None, "linestyle": None,},
]

# 图表参数
title = "XRD patterns"
xlabel = r"2$\theta$(deg.)"
ylabel = r"$\rm{Intensity\;(counts)}$"
labels = [r"$perovskite\;\rm{ABO_3}$", r"$perovskite\;\rm{A_2B_2O_6}$", r"$perovskite\;\rm{A_2B_2O_6}$"]
outfile = "xrd.pdf"
figsize = (8,6)
# 绘图设置
font_title = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
font_label = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
tick_config = {
    "major":{"which":"major","length":6,"width":1,"direction":"out","labelsize":16},
    "minor":{"which":"minor","length":4,"width":1,"direction":"out"}
    }
legend_loc = 'best'
legend_fontsize = 12
point_size = 5
line_width = 2
default_linestyle = '-'
show_name = True
# y-offset 参数
use_yoffset = True
use_normalize = True
offset_coef = 1.05
# 使用 LaTeX 渲染并控制字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
# 颜色池（自动分配） https://matplotlib.org/stable/users/explain/colors/colormaps.html
# auto_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
cmap = plt.get_cmap('tab10', 10)
auto_colors = [cmap(i) for i in range(cmap.N)]
auto_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'x', '+']

def auto_scan_files(extensions=['.xy', '.csv', '.txt', '.dat', '']):
    """
    自动扫描当前文件夹中的数据文件，并生成默认配置。
    """
    print("未检测到文件输入，正在自动扫描当前文件夹...")
    
    file_list = []
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in os.listdir(script_dir):
        # 排除隐藏文件和文件夹
        if filename.startswith('.'):
            continue
        
        # 检查文件扩展名
        _, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            file_list.append({
                "filename": filename,
                "col_x": 0,
                "col_y": 1,
                "label": os.path.splitext(filename)[0],
                "type": "line",
                "color": None,
                "marker": None,
                "linestyle": None,
            })
    
    if not file_list:
        print("未找到任何符合条件的数据文件。")
    else:
        print(f"找到以下文件: {[f['filename'] for f in file_list]}")

    return file_list

# ==================== 功能实现 ====================
# 如果files列表为空，则进行自动扫描
if not files:
    files = auto_scan_files()
    if not files:
        exit() # 如果没有找到文件，退出脚本

# ===========================================

def read_data(filename, comment_char="#", n_lines=10):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # 预读几行
    sample_lines = []
    with open(filename, 'r', newline='') as f:
        for i in range(n_lines):
            line = f.readline()
            if not line:  # 文件提前结束
                break
            sample_lines.append(line.rstrip("\r\n"))

    sample = "\n".join(sample_lines)

    # 统计注释行
    comment_lines = [line for line in sample_lines if line.strip().startswith(comment_char)]
    n_comment_lines = len(comment_lines)

    # 自动检测分隔符
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter
        use_whitespace = False
    except csv.Error:
        delimiter = r"\s+"   # 回退到任意空白
        use_whitespace = True

    # 判断首个非注释行是否为表头
    header_line_index = n_comment_lines
    header_tokens = sample_lines[header_line_index].split(delimiter if not use_whitespace else None)

    def is_number(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    if all(not is_number(tok) for tok in header_tokens):  
        header_option = 0  # 第一行是表头
    else:
        header_option = None  # 没有表头

    # 用 pandas 读数据
    return pd.read_csv(
        filename,
        sep=delimiter if not use_whitespace else delimiter,
        comment=comment_char,
        engine="python",
        header=header_option
    )

plt.figure(figsize=figsize)
ax = plt.gca()

yoffset_counter = 0
for idx, f in enumerate(files):
    try:
        data = read_data(f["filename"])
    except FileNotFoundError as e:
        print(e)
        continue

    if f["col_x"] >= data.shape[1] or f["col_y"] >= data.shape[1]:
        print(f"警告: 文件 {f['filename']} 的列索引超出范围，跳过绘图")
        continue

    x = data.iloc[:, f["col_x"]]
    y = data.iloc[:, f["col_y"]]

    # === 归一化操作（在偏移前） ===
    if use_normalize == True:
        y = y / abs(y.max())

    # === y-offset 堆积（在归一化后） ===
    if use_yoffset == True:
        # 首次进入时计算偏移量，或者重新计算
        offset_value = y.max() * offset_coef    
        y = y + yoffset_counter * offset_value
        yoffset_counter += 1

    color = f["color"] if f["color"] else auto_colors[idx % len(auto_colors)]
    linestyle = f["linestyle"] if f["linestyle"] else default_linestyle
    marker = f["marker"] if f["marker"] else auto_markers[idx % len(auto_markers)]
    if len(labels) == len(files):
        label=labels[idx]
    else:
        label = f["label"] if f["label"] else os.path.basename(f["filename"])

    if f["type"] == "scatter":
        ax.scatter(x, y, s=point_size, c=color, label=label, marker=marker)
    elif f["type"] == "line":
        ax.plot(x, y, linestyle=linestyle, linewidth=line_width, c=color, label=label)
    elif f["type"] == "line_scatter":
        ax.plot(x, y, linestyle=linestyle, linewidth=line_width, c=color, label=label, marker=marker)
        ax.scatter(x, y, s=point_size, c=color)
    else:
        print(f"未识别的绘图类型: {f['type']}, 使用散点图绘制")
        ax.scatter(x, y, s=point_size, c=color, label=label)

    if show_name == True:
    # 末点坐标（对应绘图坐标系）
        y0 = (max(y) + min(y)) * 0.5 
        x0 = (max(x) - min(x)) * 0.8
        plt.annotate(label,
                    xy=(x0, y0),
                    xytext=(x0, y0),
                    textcoords='data',
                    fontsize=16,
                    color=color,
                    ha='left', va='center',)
        
ax.set_title(title, fontdict=font_title)
ax.set_xlabel(xlabel, fontdict=font_label)
ax.set_ylabel(ylabel, fontdict=font_label)
ax.tick_params(**tick_config['major'])
ax.tick_params(**tick_config['minor'])
#ax.minorticks_on()
if show_name != True:
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
ax.grid(True, linestyle="--", alpha=0.6)
#plt.show()
plt.savefig(outfile)