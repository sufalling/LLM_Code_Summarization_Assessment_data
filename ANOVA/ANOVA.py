# 获取工作目录
import os
import sys
pwd = os.getcwd()
print(pwd)
os.chdir('E:\\learn\\PycharmProject\\bli')
pwd = os.getcwd()
print(pwd)
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List,Dict
# 正态分布
import scipy.stats as stats
from statsmodels.formula.api import ols
# 方差齐性
from scipy.stats import levene
# from scipy.stats import bartlett
# ANOVA
import pingouin as pg
# 画图
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy.ma as ma
# 保存日志
from loguru import logger

## 所有对象都用add添加，不会修改默认配置，如果修改请先logger.remove()
logger.add(pwd + "\\ANOVA\\log\\logs.log", level="DEBUG", format="{time}  |{level} \n {message}")
logger.info('start')


class ANOVA(object):

    def __init__(self,data):
        self.data:pd.DataFrame = data
        self.normal_test_output:Dict = None
        self.levene_test_output:Dict = None
        self.oneway_stat_records:pd.DataFrame = None
        self.twoway_stat_records:pd.DataFrame = None
        self.oneway_post_compare_records:pd.DataFrame = None
        self.twoway_post_compare_records:List[pd.DataFrame] = None


    def normal_test(self,class_var_name:List[str], dependent_var_name:str):

        logger.info('开始normal_test')
        normal = 0
        not_normal = 0
        not_normal = 0
        normal = 0
        if len(class_var_name) == 1:
            ## 单因素
            logger.info('开始单因素方差分析前的normal_test')
            grouped_data = self.data.groupby(class_var_name[0])
        elif len(class_var_name) > 1:
            ## 多因素无交互影响
            logger.info('开始多因素无交互作用方差分析前的normal_test')
            self.combine_two_class_var(class_var_name)
            grouped_data = self.data.groupby('combo')

        for name, group in grouped_data:# 分好组的双层列表
            # 正态检验
            ## K-S检验，样本量小不敏感，样本量大敏感，小于50个样本时选择
            # ks_stat,ks_p = stats.kstest(group[dependent_var_name].values, 'norm')
            ## W检验(Shapiro—Wilk test)更稳健
            shapiro_stat,shapiro_p = stats.shapiro(group[dependent_var_name].values)

            # if ks_p > shapiro_p:
            #     pvalue = ks_p
            #     statistic = ks_stat
            #     method = 'K-S test'
            # else:
            pvalue = shapiro_p
            statistic = shapiro_stat
            method = 'Shapiro—Wilk test'

            ## 记录满足正态分布的组数
            if pvalue > 0.05:
                logger.info(f'组{name}:经{method}，符合正态分布，pvalue={pvalue},statistic={statistic}')
                # 记录
                logger.info(f"group_name = {name}, test_name ='normal_test', method = {method}, "
                            f"statistic = {statistic}, pvalue = {pvalue},normal_test_output = success ,effect_size = Null]")
                normal = normal + 1
            else:
                logger.info(f'组{name}:经Shapiro—Wilk检验，不符合正态分布')
                # 记录
                logger.info(f"group_name = {name}, test_name ='normal_test', method = {method}, "
                            f"statistic = {statistic}, pvalue = {pvalue},normal_test_output = fail ,effect_size = Null]")
                not_normal = not_normal + 1

        ## 标记最后结果
        if not_normal == 0:
            normal_test_is = 'success'
        else:
            normal_test_is = 'fail'
        self.normal_test_output = {'output':normal_test_is,
                                   'total_sample':len(self.data),
                                   'total_group':normal + not_normal,
                                   'not_normal':not_normal,
                                   'normal':normal}
        logger.info(f"最终结果：output:{normal_test_is},total_sample:{len(self.data)},total_group:{normal + not_normal},not_normal:{not_normal},normal:{normal}")

    def combine_two_class_var(self,class_var_name:List[str]):
        # 创建水平组合列
        exec_str = "self.data['" + "'].astype('str') + '_' + self.data['".join(class_var_name) + "'].astype('str')"
        self.data['combo'] = eval(exec_str)
        return 1

    def normal_test_cross_effect(self, class_var_name: List[str], dependent_var_name:str):
        pass
        # # 多因素有交互影响
        # model = ols(f'dependent_var_name ~ C(watering) + C(sunlight) + C(watering):C(sunlight)', data=self.data).fit()
        # residuals = model.resid
        # # Shapiro-Wilk检验
        # shapiro_stat, shapiro_p = stats.shapiro(residuals)
        # print(f"Shapiro-Wilk: stat={shapiro_stat:.3f}, p={shapiro_p:.4f}")
        # # p > 0.05 表示残差近似正态


    def levene_test(self,class_var_name:List[str], dependent_var_name):
        # 方差齐性检验
        # dependent_var因变量，连续变量
        levene_test_output = None
        if len(class_var_name) == 1:
            ## 单因素
            logger.info("开始单因素方差齐性检验，levene_test")
            grouped_data = self.data.groupby(class_var_name[0])
        elif len(class_var_name) > 1:
            ## 多因素无交互影响
            logger.info("开始多因素无交互作用方差齐性检验，levene_test")
            self.combine_two_class_var(class_var_name)
            grouped_data = self.data.groupby('combo')

        grouped_levene_data = [group[dependent_var_name].values for name, group in grouped_data]
        ## Python语法，*列表解包，同类还有**字典解包
        levene_stat, levene_p = levene(*grouped_levene_data, center='median')
        ## 另一种方法
            # levene_result = pg.homoscedasticity(data_func_init, dv = dependent_var,group= class_var,method = 'Levene')
        if levene_p > 0.05:
            logger.info(f'经levene检验各组方差齐性,F={levene_stat}, p={levene_p},total_group:{len(grouped_data)},'
                        f'total_sample{len(self.data)},levene_test_output = success')
            # 记录
            self.levene_test_output = {'output':'success',
                                       'total_sample':len(self.data),
                                       'total_group':len(grouped_data),
                                       'F-value':levene_stat,
                                       'p-value':levene_p}

        else:
            logger.info(f'经levene检验各组方差不是齐性的，F={levene_stat}, p={levene_p},total_group:{len(grouped_data)},'
                        f'total_sample{len(self.data)},levene_test_output = fail')
            # 记录
            self.levene_test_output = {'output': 'fail',
                                       'total_sample': len(self.data),
                                       'total_group': len(grouped_data),
                                       'F-value': levene_stat,
                                       'p-value': levene_p}
        return levene_stat, levene_p, levene_test_output

    def one_way_anova(self, class_var_name: List[str] = None, dependent_var_name: str = None):
        # 单因素ANOVA
        logger.info('开始单因素方差分析')
        if self.levene_test_output['output'] == 'success':
            ## 方差齐性使用
            logger.info('方差齐性，使用正常ANOVA')
            anova_result = pg.anova(data=self.data, dv=dependent_var_name, between=class_var_name[0], detailed=True)
            ## 结果解读：SS-sums of squares，DF-degrees of freedom，MS-Mean squares，F F-value，
            # p_unc uncorrected p-values,np2-Partial eta-square effect sizes(偏η方效应量，# 解释：0.01 小效应，0.06 中效应，0.14 大效应)
        elif self.levene_test_output['output'] == 'fail':
            ## 方差不齐时使用
            print('方差不齐，使用welch_anova')
            anova_result = pg.welch_anova(data=self.data, dv=dependent_var_name, between=class_var_name[0])
            ## 解读： ddof1-Numerator degrees of freedom,ddof2-Denominator degrees of freedom,
            # F,p_unc-uncorrected p-values,np2-Partial eta-squared
        self.oneway_stat_records = anova_result
        logger.info(anova_result)
        return anova_result

    def two_way_anova(self,class_var_name: List[str], dependent_var_name: str = None):
        # 双因素ANOVA

        logger.info('开始双因素方差分析')
        twoway_anova_result = pg.anova(data = self.data,dv=dependent_var_name,
                                        between=class_var_name)
        self.twoway_stat_records = twoway_anova_result
        logger.info(twoway_anova_result)
        return twoway_anova_result


    def oneway_post_compare(self, class_var_name: List[str] = None, dependent_var_name: str = None):
        # 确定主效应有显著差异后,事后两两比较

        logger.info('单因素方差分析的事后比较')

        if self.levene_test_output['output'] == 'success':
            logger.info("TurkeyHSD，适用于方差齐性")
            ## 若方差齐性，使用TurkeyHSD,看P-Turkey的值，Hedges effect是效应量大小,可设置
            # 'A': Name of first measurement
            # 'B': Name of second measurement
            # 'mean_A': Mean of first measurement
            # 'mean_B': Mean of second measurement
            # 'diff': Mean difference (= mean(A) - mean(B))
            # 'se': Standard error
            # 'T': T-values
            # 'p_tukey': Tukey-HSD corrected p-values
            # 'hedges'/'eta_square': Hedges effect size (or any effect size defined in effsize)
            posthoc_result = pg.pairwise_tukey(data=self.data, dv=dependent_var_name, between=class_var_name[0],
                                           effsize='eta_square')
        if self.levene_test_output['output'] == 'fail':
            ## 方差不齐性，使用Games‑Howell检验
            # 'A': Name of first measurement
            # 'B': Name of second measurement
            # 'mean_A': Mean of first measurement
            # 'mean_B': Mean of second measurement
            # 'diff': Mean difference (= mean(A) - mean(B))
            # 'se': Standard error
            # 'T': T-values
            # 'df': adjusted degrees of freedom
            # 'pval': Games-Howell corrected p-values
            # 'hedges': Hedges effect size (or any effect size defined in effsize)
            #
            logger.info("Games‑Howell检验，适用于方差不一致")
            posthoc_result = pg.pairwise_gameshowell(data=self.data, dv=dependent_var_name, between=class_var_name[0],
                                                 effsize='eta_square')

        self.oneway_post_compare_records = posthoc_result
        logger.info(posthoc_result)
        return posthoc_result

    def twoway_post_compare(self, class_var_name: List[str] = None, dependent_var_name: str = None):

        logger.info('多因素无交互作用方差分析的主效应事后比较')
        if self.levene_test_output['output'] == 'success':
            # 方差齐性
            logger.info(f"{class_var_name[0]}主效应事后检验,TurkeyHSD，适用于方差齐性")
            posthoc_result1 = pg.pairwise_tukey(data=self.data, dv=dependent_var_name, between=class_var_name[0],
                                               effsize='eta_square')

            logger.info(f"\n{class_var_name[1]}主效应事后检验,TurkeyHSD，适用于方差齐性")
            posthoc_result2 = pg.pairwise_tukey(data=self.data, dv=dependent_var_name, between=class_var_name[1],
                                                effsize='eta_square')

        elif self.levene_test_output['output'] == 'fail':
            ## 方差不齐性，使用Games‑Howell检验
            logger.info(f"\n {class_var_name[0]}主效应事后检验,Games‑Howell检验，适用于方差不一致")
            posthoc_result1 = pg.pairwise_gameshowell(data=self.data, dv=dependent_var_name, between=class_var_name[0],
                                                     effsize='eta_square')
            logger.info(f"\n {class_var_name[1]}主效应事后检验,Games‑Howell检验，适用于方差不一致")
            posthoc_result2 = pg.pairwise_gameshowell(data=self.data, dv=dependent_var_name, between=class_var_name[1],
                                                     effsize='eta_square')

        logger.info(posthoc_result1)
        logger.info(posthoc_result2)
        self.twoway_post_compare_records = [posthoc_result1, posthoc_result2]
        return (posthoc_result1,posthoc_result2)

    def post_compare_picture(self,data, name_p, save_path):
        # ------------------------------
        # 1. 数据透视，得到 T 统计量和 P 值的矩阵
        # ------------------------------
        pivot_t = data.pivot_table(index='A', columns='B', values='T', aggfunc='mean')
        pivot_p = data.pivot_table(index='A', columns='B', values=name_p, aggfunc='mean')

        # 提取行列索引，并转换为 numpy 数组
        rows = pivot_t.index.tolist()
        cols = pivot_t.columns.tolist()
        t_matrix = pivot_t.values
        p_matrix = pivot_p.values

        # ------------------------------
        # ---------- 2. 颜色映射：深蓝色（P<0.05）vs 浅灰色（P≥0.05）----------
        color_val = np.zeros_like(p_matrix)
        color_val[p_matrix < 0.05] = 1  # 1 -> 深蓝色，0 -> 浅灰色
        mask = np.isnan(p_matrix)
        data = np.ma.masked_where(mask, color_val)

        # 自定义颜色映射：0=青色, 1=红色, bad=白色
        cmap = ListedColormap(['#D3D3D3', '#1f77b4'])  # 浅灰色（#D3D3D3）和 深蓝色（#1f77b4）
        cmap.set_bad(color='white')

        # ------------------------------
        # 3. 绘图
        # ------------------------------
        # 动态尺寸
        n_rows, n_cols = len(rows), len(cols)
        fig_width = max(10, n_cols * 1)
        fig_height = max(8, n_rows * 0.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        # 设置坐标轴刻度与标签
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(rows)))
        ax.set_xticklabels(cols, rotation=45, fontsize=9, ha='right')
        ax.set_yticklabels(rows, rotation=45, ha='right', fontsize=9)
        # ax.set_xlabel('类别2', fontsize=11)
        # ax.set_ylabel('类别1', fontsize=11)

        # ------------------------------
        # 4. 添加文本：T统计量 + 上标星号
        # ------------------------------
        # 内部文字动态字体
        fontsize_text = max(10, min(8, 80 // max(n_cols, n_rows)))

        for i in range(n_rows):
            for j in range(n_cols):
                t_val = t_matrix[i, j]
                p_val = p_matrix[i, j]
                if np.isnan(t_val) or np.isnan(p_val):
                    continue

                if p_val < 0.001:
                    stars = '***'
                elif p_val < 0.01:
                    stars = '**'
                elif p_val < 0.05:
                    stars = '*'
                else:
                    stars = ''

                if stars:
                    text = f"${t_val:.2f}^{{{stars}}}$"
                else:
                    text = f"{t_val:.2f}"

                # 添加白色半透明背景防止文字与背景色混淆
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=fontsize_text, color='black',
                        rotation=45)  # 逆时针旋转45度，实现向左上倾斜效果

        # ------------------------------
        # 5. 添加图例
        # ------------------------------
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', edgecolor='black', label='P < 0.05'),
            Patch(facecolor='#D3D3D3', edgecolor='black', label='P ≥ 0.05')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()


    def report(self, class_var_name: List[str] = None, dependent_var_name: str = None):

        logger.info(f"""\n============== 报告汇总 ======================
                    影响因素{class_var_name}，因变量{dependent_var_name}""")
        if self.normal_test_output:
            logger.info(f"""\n================== 正态检验 ======================
                    {self.normal_test_output}""")
        if self.levene_test_output:
            logger.info(f"""\n================== 方差齐性检验 ===================
                    {self.levene_test_output}""")
        logger.info("""\n=================== 方差分析 =====================""")

        if self.oneway_stat_records is not None:
            logger.info(f"""\n=================== 1.单因素方差分析：影响因素{class_var_name}，因变量{dependent_var_name} ===============
                    {self.oneway_stat_records}""")

        if self.twoway_stat_records is not None:
            logger.info(f"""\n=============== 2.双因素方差分析：影响因素{class_var_name}，因变量{dependent_var_name} =============
                    {self.twoway_stat_records}""")
        logger.info(f"""\n=================== 事后比较 =====================""")
        if self.oneway_post_compare_records is not None:
            logger.info(f"""\n============= 1.单因素方差分析的事后比较：影响因素{class_var_name}，因变量{dependent_var_name}==============""")
            logger.info("""\n 1.1 保存数据表""")
            ## 1.1 保存
            os.makedirs(os.path.dirname(
                f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\oneway_post_compare_result.csv"
            ), exist_ok=True)
            self.oneway_post_compare_records.to_csv(
                f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\oneway_post_compare_result.csv"
                , sep='\t', index=False)
            logger.info("""\\n 1.2 保存热力图""")
            if self.levene_test_output['output'] == 'success':
                self.post_compare_picture(self.oneway_post_compare_records,name_p='p_tukey',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap.png")
            elif self.levene_test_output['output'] == 'fail':
                self.post_compare_picture(self.oneway_post_compare_records,name_p='pval',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap.png")

        if self.twoway_post_compare_records is not None:
            logger.info(f"""\n============= 2.双因素无交互方差分析的事后比较：影响因素{class_var_name}，因变量{dependent_var_name}=============""")
            logger.info("""\n 1.1 保存数据表""")
            os.makedirs(os.path.dirname(
                f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\twoway_post_compare_result1.csv"
            ), exist_ok=True)
            self.twoway_post_compare_records[0].to_csv(
                f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\twoway_post_compare_result_{class_var_name[0]}.csv"
                , sep='\t', index=False)
            self.twoway_post_compare_records[1].to_csv(
                f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\twoway_post_compare_result——{class_var_name[1]}.csv"
                , sep='\t', index=False)
        
            logger.info("""\n 1.2 保存图表""")
            if self.levene_test_output['output'] == 'success':
                self.post_compare_picture(self.twoway_post_compare_records[0], 'p_tukey',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap_{class_var_name[0]}.png")
                self.post_compare_picture(self.twoway_post_compare_records[1], 'p_tukey',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap_{class_var_name[1]}.png")
            elif self.levene_test_output['output'] == 'fail':
                self.post_compare_picture(self.twoway_post_compare_records[0], name_p='pval',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap_{class_var_name[0]}.png")
                self.post_compare_picture(self.twoway_post_compare_records[1], name_p='pval',
                                          save_path=f"{pwd}\\ANOVA\\save\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap_{class_var_name[1]}.png")



    def exec_anova(self,class_var_name:List[str], dependent_var_name:str):
        self.normal_test(class_var_name, dependent_var_name)
        self.levene_test(class_var_name, dependent_var_name)
        if len(class_var_name) == 1:
            self.one_way_anova(class_var_name, dependent_var_name)
            self.oneway_post_compare(class_var_name, dependent_var_name)
        elif len(class_var_name) == 2:
            self.two_way_anova(class_var_name, dependent_var_name)
            self.twoway_post_compare(class_var_name, dependent_var_name)
        self.report(class_var_name, dependent_var_name)



if __name__ == '__main__':
    # 读取数据
    # data_func_init = pd.read_excel(pwd + '\\ANOVA\\all_0.3-2.xlsx', sheet_name='func',usecols=['model', 'reponame', 'field', 'prompt', 'LLM_as_judge',
    #        'sentenceTransformer_similarity', 'meteor_score','sacrebleu_bleu_4', 'rouge_l_f'])
    # data_func_init = pd.concat([data_func_init,data_func_init,data_func_init])
    # # data_func_init = data_func_init[data_func_init['reponame'] == 'jsoncpp']

    data_module_init = pd.read_excel(pwd + '\\ANOVA\\all_0.3_prompt8_qwen.xlsx', sheet_name='module',
                                     usecols=['reponame', 'field', 'LLM_as_judge',
                                              'sentenceTransformer_similarity', 'meteor_score', 'sacrebleu_bleu_4',
                                              'rouge_l_f'])
    data_module_init = pd.concat(
        [data_module_init, data_module_init, data_module_init, data_module_init])

    # data_repo_init = pd.read_excel(pwd + '\\ANOVA\\all_0.3-2.xlsx', sheet_name='repo')
    # data_repo_init = pd.concat([data_repo_init, data_repo_init, data_repo_init, data_repo_init, data_repo_init])
    # data_repo_init = pd.concat([data_repo_init, data_repo_init, data_repo_init, data_repo_init])
    # data_repo_init = pd.concat([data_repo_init, data_repo_init])
    #
    # anova = ANOVA(data_module_init)
    # anova.exec_anova(class_var_name=['reponame'], dependent_var_name=metric)
    for metric in ['LLM_as_judge',
           'sentenceTransformer_similarity', 'meteor_score','sacrebleu_bleu_4', 'rouge_l_f']:
        anova = ANOVA(data_module_init)
        anova.exec_anova(class_var_name=['reponame'], dependent_var_name=metric)# 顺序model,prompt


        # anova = ANOVA(data_repo_init)
        # anova.exec_anova(class_var_name=['reponame'], dependent_var_name=metric)
        #
        #




    # # test
    # data_func_init = pd.concat([data_func_init,data_func_init,data_func_init])
    # data_func_init['prompt'] = data_func_init['prompt'].astype('int32')
    # data_func_init['model'] = data_func_init['model'].astype('str')
    # data_func_init['comb'] = data_func_init['model'].astype('str') + '_' + data_func_init['prompt'].astype('str')
    # test_class_var_name = ['model','prompt']
    # test_str =  "data_func_init['" + "'].astype('str') + '_' + data_func_init['".join(test_class_var_name) + "'].astype('str')"
    # data_func_init['comb'] = eval(test_str)
    # data_func_init
    # del data_func_init['comb']
    # comb = data_func_init.groupby('comb')
    # for name, group in comb:
    #     print(name)
    #     print(group['LLM_as_judge'].values)
    # ## K-S检验，样本量小不敏感，样本量大敏感，小于50个样本时选择
    # stst,p = stats.kstest(data_func_init['LLM_as_judge'],'norm')
    # kstest(data_func_init['LLM_as_judge'],'norm').statistic
    # kstest(data_func_init['LLM_as_judge'],'norm').pvalue
    # ## W检验(Shapiro—Wilk test)更稳健
    #
    # stat,p = shapiro(data_func_init['LLM_as_judge'])
    # shapiro(data_func_init['LLM_as_judge']).pvalue
    #
    # unique_values = data_func_init['model'].unique()
    # levene_stat, lenven_p=levene(data_func_init.LLM_as_judge,
    #        data_func_init.model,center='median')
    #
    # grouped_data = [group['LLM_as_judge'].values for name, group in data_func_init.groupby('model')]
    # levene_stat, lenven_p=levene(*grouped_data,center='median')
    #
    # anova_result = pg.anova(data= data_func_init,dv='LLM_as_judge',between='model',detailed=True)
    # anova_result = pg.welch_anova(data= data_func_init,dv='LLM_as_judge',between='model')
    # anova_result_prompt = pg.welch_anova(data= data_func_init,dv='LLM_as_judge')
    #
    #
    # two_way_anova_result = pg.anova(data = data_func_init,dv="LLM_as_judge", between=["model", "prompt"])
    # print(two_way_anova_result)
    # anova_result.F[0]
    # logger.info(anova_result)
    # posthoc_result = pg.pairwise_tukey(data_func_init,dv='LLM_as_judge',between='prompt',effsize='eta_square')
    # posthoc2_result = pg.pairwise_gameshowell(data_func_init,dv='LLM_as_judge',between='model',effsize='eta_square')
    # logger.info({'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6})
    #
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    # # 拟合双因素模型
    # models = ols('LLM_as_judge ~ C(model) * C(prompt)', data=data_func_init).fit()
    # # 计算异方差稳健标准误（HC3 通常推荐）
    # robust_result = models.get_robustcov_results(cov_type='HC3')
    # print(robust_result)
    # print(robust_result.summary())  # 查看 F 检验和 p 值
    #
    # f"{pwd}\\ANOVA\\save\\{'_'.join(['1','2'])}_3\\oneway_anova_result.csv"
    #
    #
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    # import numpy.ma as ma
    # def post_compare_picture(data,name_p, class_var_name: List[str] = None, dependent_var_name: str = None):
    #     # ------------------------------
    #     # 1. 数据透视，得到 T 统计量和 P 值的矩阵
    #     # ------------------------------
    #     pivot_t = data.pivot_table(index='A', columns='B', values='T', aggfunc='mean')
    #     pivot_p = data.pivot_table(index='A', columns='B', values=name_p, aggfunc='mean')
    #
    #     # 提取行列索引，并转换为 numpy 数组
    #     rows = pivot_t.index.tolist()
    #     cols = pivot_t.columns.tolist()
    #     t_matrix = pivot_t.values
    #     p_matrix = pivot_p.values
    #
    #
    #
    #     # ------------------------------
    #     # ---------- 2. 颜色映射：深蓝色（P<0.05）vs 浅灰色（P≥0.05）----------
    #     color_val = np.zeros_like(p_matrix)
    #     color_val[p_matrix < 0.05] = 1  # 1 -> 深蓝色，0 -> 浅灰色
    #     mask = np.isnan(p_matrix)
    #     data = np.ma.masked_where(mask, color_val)
    #
    #     # 自定义颜色映射：0=青色, 1=红色, bad=白色
    #     cmap = ListedColormap(['#D3D3D3', '#1f77b4'])  # 浅灰色（#D3D3D3）和 深蓝色（#1f77b4）
    #     cmap.set_bad(color='white')
    #
    #
    #     # ------------------------------
    #     # 3. 绘图
    #     # ------------------------------
    #     # 动态尺寸
    #     n_rows, n_cols = len(rows), len(cols)
    #     fig_width = max(10, n_cols * 1)
    #     fig_height = max(8, n_rows * 0.8)
    #     fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    #     im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    #
    #     # 设置坐标轴刻度与标签
    #     ax.set_xticks(np.arange(len(cols)))
    #     ax.set_yticks(np.arange(len(rows)))
    #     ax.set_xticklabels(cols, rotation=45, fontsize=9, ha='right')
    #     ax.set_yticklabels(rows, rotation=45, ha='right',fontsize=9)
    #     # ax.set_xlabel('类别2', fontsize=11)
    #     # ax.set_ylabel('类别1', fontsize=11)
    #
    #     # ------------------------------
    #     # 4. 添加文本：T统计量 + 上标星号
    #     # ------------------------------
    #     # 内部文字动态字体
    #     fontsize_text = max(10, min(8, 80 // max(n_cols, n_rows)))
    #
    #     for i in range(n_rows):
    #         for j in range(n_cols):
    #             t_val = t_matrix[i, j]
    #             p_val = p_matrix[i, j]
    #             if np.isnan(t_val) or np.isnan(p_val):
    #                 continue
    #
    #             if p_val < 0.001:
    #                 stars = '***'
    #             elif p_val < 0.01:
    #                 stars = '**'
    #             elif p_val < 0.05:
    #                 stars = '*'
    #             else:
    #                 stars = ''
    #
    #             if stars:
    #                 text = f"${t_val:.2f}^{{{stars}}}$"
    #             else:
    #                 text = f"{t_val:.2f}"
    #
    #             # 添加白色半透明背景防止文字与背景色混淆
    #             ax.text(j, i, text, ha='center', va='center',
    #                     fontsize=fontsize_text, color='black',
    #                     rotation=45)  # 逆时针旋转45度，实现向左上倾斜效果
    #
    #     # ------------------------------
    #     # 5. 添加图例
    #     # ------------------------------
    #     from matplotlib.patches import Patch
    #     legend_elements = [
    #         Patch(facecolor='#1f77b4', edgecolor='black', label='P < 0.05'),
    #         Patch(facecolor='#D3D3D3', edgecolor='black', label='P ≥ 0.05')
    #     ]
    #     ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    #
    #     plt.tight_layout()
    #
    #     # 保存
    #     save_path = f"{pwd}\\ANOVA\\test\\{'_'.join(class_var_name)}_{dependent_var_name}\\heatmap.png"
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.show()
    #
    # post_compare_picture(posthoc_result,'p_tukey',['model','prompt'],'LLM_as_judge')
