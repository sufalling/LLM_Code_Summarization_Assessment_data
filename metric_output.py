# -*- coding:utf-8 -*-
import copy
import time
from openai import OpenAI
from datasets import load_from_disk, Dataset, concatenate_datasets, load_dataset
import string
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from dataclasses import dataclass
import os


from torch.xpu import device

@dataclass
class Setting(object):
    # 设置LLM_as_judge评估指标使用的API平台、模型、超时设置、温度和top_p设置,研究报告的实验使用来自第三方平台的gpt-4o
    LLM_as_judge_API_platform: str = 'qwen'
    LLM_as_judge_model: str = 'qwen-plus'
    LLM_as_judge_timeout: int = 30
    LLM_as_judge_temperature: float = 1
    LLM_as_judge_top_p: float = 1

    # 设置语义相似度指标使用的模型，默认all-MiniLM-L6-v2; 使用CPU或cuda
    # device – Device (like "cuda", "cpu", "mps", "npu") that should be used for computation.
    # If None, checks if a GPU can be used.
    SentenceTransformer_model: str = 'all-MiniLM-L6-v2'
    sentence_transformer_device = 'cpu'
    # 初始化语义相似度指标使用的模型类
    sentence_model = SentenceTransformer(SentenceTransformer_model, device=sentence_transformer_device)

    # 初始化计算Rouge的类
    rouge = Rouge()

    # 设置使用哪个大语言模型生成文档，以及温度等设置
    API_platform:str = 'qwen'
    model: str = 'qwen-plus'
    temperature: float = 1
    top_p: float = 1


class Evaluation(object):

    # 初始化放入类的各项大语言模型设置、输入的数据集、默认数据集中人工注释为'comment'列，大语言模型生成的注释为'detail_output'列
    # 可以改变默认要输入的两列的名称
    # 数据集输入前请保证至少有'comment' 和'detail_output'列（名字可以在初始化Evaluation类时修改成你数据集匹配的名称）
    def __init__(self, settings: Setting, input_data: Dataset = None, generation_doc: str = 'detail_output',comment: str = 'comment',times: int = 1):
        self.setting = settings
        self.generation_doc = generation_doc
        self.comment = comment
        self.data = input_data
        # self.rouge = Rouge()
        # 加载需要的API
        self.api_and_url = Evaluation.get_api_and_url("./configure.txt")
        self.sentence_model = self.setting.sentence_model
        self.times = times

    # 读取给出的大语言模型base_url和API_KEY
    #
    # 格式: 平台名_base_url:https://xxxxx.xxxxx
    #      平台名_API_KEY:xxxxxx
    # 不同平台其格式有差别，此处把常用平台的都放过来，只需要填好API_key就可
    #
    # 注意不要加任何空行！！！


    @staticmethod
    def get_api_and_url(configure_path: str):
        dict_api_url = {}
        with open(configure_path) as f:
            text = f.readlines()
            for item in text:
                # 如果为空行，跳过
                if not item:
                    continue
                item = item.strip("\n").split(":")
                if 'url' in item[0]:
                    dict_api_url[item[0]] = "https:" + item[2]
                elif "xunfei" in item[0]:
                    if "deepseek" not in item[0]:
                        dict_api_url[item[0]] = item[1] + ":" + item[2]
                    else:
                        dict_api_url[item[0]] = item[1]
                else:
                    dict_api_url[item[0]] = item[1]
        return dict_api_url

    # 加载数据集，转化为huggingface包的Dataset格式
    @staticmethod
    def upload(code_ds: str, split: str = 'test', begin_num: int = 1, end_num: int = 0, sheet_name: str = None):
        """
        :param code_ds: 数据集路径
        :param split: 加载数据集下的train或test
        :param begin_num: 加载位置
        :param end_num: 加载位置
        :param sheet_name: xlsx，选择加载的工作表
        :return Dataset
        """
        if 'csv' in code_ds:
            datas = load_dataset("csv", data_files=f"{code_ds}")
            # 读进来默认自动分成一个train
            ds_tests = datas['train']
        elif 'tsv' in code_ds or 'txt' in code_ds:
            datas = load_dataset("csv", data_files=f"{code_ds}", sep='\t')
            # 读进来默认自动分成一个train
            ds_tests = datas['train']
        elif 'xlsx' in code_ds:
            df = pd.read_excel(f"{code_ds}", sheet_name=sheet_name)
            ds_tests = Dataset.from_pandas(df)
        else:
            try:
                datas = load_from_disk(f"{code_ds}")
                ds_tests = datas[split]
            except:
                print("加载Arrow文件失败")
        if end_num != 0:
            ds_tests = ds_tests.select(range(begin_num - 1, end_num - 1))
        return ds_tests

    # 对比前去掉标点
    @staticmethod
    def remove_punctuation(sentence: str):
        for i in string.punctuation:
            sentence = sentence.replace(i, "")
        sentence = sentence.replace('\n', '')
        return sentence

    def add_sacrebleu_bleu_4(self,references: str, candidate_list: str):
        # BLEU-4,0-1
        # print('bleu')
        try:
            # references要变成列表，列表里是一个个句子
            bleu_4 = sacrebleu.sentence_bleu(candidate_list, [references])
            # print(bleu_4.score)
            return bleu_4.score / 100
        except:
            return 'None_output'

    def add_meteor(self,references, candidate_list):
        try:
            # bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            # Meteor,https://github.com/zembrodt/pymeteor
            # 用nltk实现
            # print('meteor')
            meteor_score_out = meteor_score(references, candidate_list)  # 输入分词后的列表
            return float(meteor_score_out)
        except:
            return 'None_output'

    def add_rouge(self,references, candidate):

        # 一般用rouge的F值
        try:
            # 输入句子

            rouge_scores = self.setting.rouge.get_scores(candidate, references, avg=True)
            return rouge_scores
        except:
            return {"rouge-1": {'f': 0, 'p': 0, 'r': 0}, "rouge-2": {'f': 0, 'p': 0, 'r': 0},
                    "rouge-l": {'f': 0, 'p': 0, 'r': 0}}

    def add_metrics(self,example):


        hypothesis = self.remove_punctuation(example[self.generation_doc])
        references = self.remove_punctuation(example[self.comment])

        reference = [word_tokenize(references)]  ## 分词,reference已经是一个套了两层的列表
        candidate = word_tokenize(hypothesis)

        sacrebleu_bleu_score = self.add_sacrebleu_bleu_4(self.comment, self.generation_doc)
        meteor_score_out = self.add_meteor(reference, candidate)
        rouge_scores = self.add_rouge(references, hypothesis)
        # 'bleu_1': bleu_score[0],'bleu_2':bleu_score[1],'bleu_3':bleu_score[2],
        return {'meteor_score': meteor_score_out,
                'sacrebleu_bleu_4': sacrebleu_bleu_score,
                'rouge_1_f': rouge_scores["rouge-1"]['f'], 'rouge_1_p': rouge_scores["rouge-1"]['p'],
                'rouge_1_r': rouge_scores["rouge-1"]['r'],
                'rouge_2_f': rouge_scores["rouge-2"]['f'], 'rouge_2_p': rouge_scores["rouge-2"]['p'],
                'rouge_2_r': rouge_scores["rouge-2"]['r'],
                'rouge_l_f': rouge_scores["rouge-l"]['f'], 'rouge_l_p': rouge_scores["rouge-l"]['p'],
                'rouge_l_r': rouge_scores["rouge-l"]['r'],
                }

    def set_llm_as_judge(self,example, temperature: float = 1, top_p: float = 1):
        # example是一个字典
        client = OpenAI(
            api_key=self.api_and_url[f'{self.setting.LLM_as_judge_API_platform}_API_KEY'],
            base_url=self.api_and_url[f'{self.setting.LLM_as_judge_API_platform}_base_url']
        )
        time.sleep(0.5)
        print('set-llm-as-judge')
        try:
            chat_completion = client.chat.completions.create(
                model=self.setting.LLM_as_judge_model,
                messages=[
                    # 角色设定
                    {
                        "role": "system",
                        "content": "You are Frederic, a veteran code expert with a deep understanding of what code does in a project. Your task is to Compare the similarity of two code comments."
                    },
                    # 用户
                    {
                        "role": "user",
                        "content": f"Compare the similarity score between {example[self.generation_doc]} and {example[self.comment]}, The second code snippet serves as a reference. Comparisons can be made from the following aspects: \n - Whether the word expression is the same \n- Whether similar function descriptions are accurate \n- Whether the functional logic is similar and reasonable \nNow please give a score of 0-100, where 0 means the contents are not similar at all and 100 means the contents are completely similar. \n**note**: Just output the score. Don't explain."
                    }
                ],
                # 介于 0 和 2 之间。较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更加集中和确定,默认为1
                temperature=self.setting.LLM_as_judge_temperature,
                # 默认为1，范围-1~1
                top_p=self.setting.LLM_as_judge_top_p
            )
            return {'LLM_as_judge': int(chat_completion.choices[0].message.content) / 100}
        except Exception as e:
            print(e)
            return {'LLM_as_judge': 0}

    def generate_embedding(self,example):

        # Download model
        # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print('embedding')
        comment = self.remove_punctuation(example[self.comment])
        model_output = self.remove_punctuation(example[self.generation_doc])
        try:
            # Get embeddings of sentences
            embeddings_comment = self.sentence_model.encode(comment, convert_to_tensor=True)
            embeddings_comment = embeddings_comment.to("cuda")

            embeddings_model_output = self.sentence_model.encode(model_output, convert_to_tensor=True)
            embeddings_model_output = embeddings_model_output.to("cuda")

            # calculate similarity
            cosine_scores = util.cos_sim(embeddings_comment, embeddings_model_output)
            return {"sentenceBERT_similarity": float(cosine_scores)}
        except Exception as e:
            print({e})
            return {"sentenceBERT_similarity": 'None_output'}

    # 按模型保存数据集到单个文件
    @staticmethod
    def save(past_ds: Dataset, save_path: str = './data'):
        past_ds.to_csv(path_or_buf=f'{save_path}.txt', sep='\t')
        past_df = past_ds.to_pandas()
        past_df.to_excel(f'{save_path}.xlsx',index=False)

    def eva(self,save_path='./data'):
        # 依次执行前面的评估
        for i in range(self.times):
            datas = copy.deepcopy(self.data.map(self.set_llm_as_judge)) # map后的数据集要保存
            datas = datas.map(self.generate_embedding)
            datas = datas.map(self.add_metrics)
            datas = datas.to_pandas()
            if i == 0:
                data_temp:Dataset = copy.deepcopy(datas)
            else:
                data_temp = pd.concat([data_temp,datas])

        dirname = os.path.dirname(save_path)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        data_temp = Dataset.from_pandas(data_temp)
        Evaluation.save(data_temp,save_path)



if __name__ == '__main__':
    # 配置大语言模型，其他默认
    setting1 = Setting(SentenceTransformer_model='cuda',LLM_as_judge_API_platform='aihubmix',LLM_as_judge_model='gpt-4o-2024-11-20')
    # 读取数据
    # data0 = Evaluation.upload("./testcases_official_doc.xlsx", sheet_name='function')
    data1 = Evaluation.upload("./testcases_official_doc.xlsx",sheet_name='module')
    data2 = Evaluation.upload("./testcases_official_doc.xlsx",sheet_name='repo')

    # 初始化
    # evaluation0 = Evaluation(setting1, data0, 'model_output', 'comment')
    # 模块20次
    evaluation1 = Evaluation(setting1, data1,'generate_module','manual_module',times=1)
    # 仓库40次
    evaluation2 = Evaluation(setting1, data2,'repo_desc','about',times=1)

    # 评估并保存
    # evaluation0.eva(save_path='./tools_metrics_output/function')
    evaluation1.eva(save_path='./tools_metrics_output/module')
    evaluation2.eva(save_path='./tools_metrics_output/repo')




