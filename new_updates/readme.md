## Updates

[2023.1.12]: We integrate our model with the newly released [FinGPT-Forecaster](https://github.com/AI4Finance-Foundation/FinGPT/blob/master/fingpt/FinGPT_Forecaster/README.md). Try their demo [here](https://huggingface.co/spaces/FinGPT/FinGPT-Forecaster)

  

### Overview

  

This project leverages the FinGPT-Forecaster, a state-of-the-art model designed for financial market predictions by fine-tuning Meta's Llama. The system analyzes market news and predicts stock price movements.

  

### Requirements

  

Python 3.x

PyTorch

Transformers library

Pandas

M2M100 model for translation

  

### Installation

  

1. Clone the repository:

```bash
git clone https://github.com/your-repo/stock-movement-predictor.git
```
2. Install the required packages:
[Note]: We have most of the key requirements in `requirements.txt`. Please modify based on your own installment.
```bash
pip install -r requirements.txt
  ```

### Usage

  

1. Set the news_dir and stock_dir variables to your directories containing news data and stock data, respectively.

Run the script to process the data and generate forecasts:

2. Run the script to process the data and generate forecasts:

```bash
python finGPT.py
```
[Note]: Please modify fin_data with the company of your desire. Our script assumes each file name is the ticker code of the stock you want to analyze.

  

### Features



1. Stock Data Processing: Reads and processes stock price movement data.

2. News Data Translation and Processing: Translates non-English news data and integrates it for analysis. This is because FinGPT-Forecaster is trained on [Yahoo Finance](https://finance.yahoo.com) and currently only accept English.

3. Forecast Generation: Uses the FinGPT-Forecaster model to generate stock movement predictions.

4. Output Management: Saves the forecast results to a specified file for easy access and analysis.

### Limitations
1. We think future updates can explore the integration of Google Looker Studio's API to add new features like `[Company Introduction]` and `[Basic Financials]` . This integration aims to enhance the script's functionality and provide users with more comprehensive and up-to-date financial data.
2. Due to current data limitations, we are unable to showcase specific improvements enabled by the FinGPT model. Future work will focus on evaluating this model to better understand its potential and limitations, especially in comparison to other LLMs fine-tuned for finance (e.g., GPT-4, MistralAI).
### Contribution

Feel free to fork this project, submit pull requests, or suggest improvements via issues.

## 数据准备（对应代码位于`data_preparations`文件夹，本节提到的代码文件如无特殊说明均位于该文件夹）
环境依赖：可以直接通过`pip install -r requirements.txt`来安装。
1. 更新`hs_restricted.csv`为需要的股票
2. 运行`download_titles.py`爬取新闻数据，数据会存储到`fin_data/scrape/results`，再使用`download_contents.py`下载全部内容，并使用`fin_data/clean_data.py`清洗，清洗后的数据在`fin_data/results_with_content_clean`
    * 需要使用[快代理](https://www.kuaidaili.com/)的隧道代理，注册登录后进入“隧道代理”界面，新增隧道代理后，修改`config`变量中的如下内容：
        * tunnel：<隧道host>:<HTTP端口>
        * username：<用户名>
        * password：<密码>
    * 目前代码中的相应信息为我的快代理账号，里面还有一些余额，也可以先继续使用
3. 运行`read_clickhouse_data.py`读取Clickhouse中相关新闻、公告数据，数据会存储到`fin_data/clickhouse_data`
4. 运行`add_labels.py`为每条新闻/公告添加标签及股价信息，数据存储于`fin_data/content_with_labels`
    * 可通过调控THRESH_VERY、THRESH、FORWARD_DAYS、END_DATE进行调控
    * 代码使用akshare读取从2012年1月1日到END_DATE之间的股票信息
    * 使用UnivariateSpline拟合股价曲线，如果希望观察拟合曲线可以运行`fin_data/candle_figure.py`查看
    * 标签定为新闻/公告发布后最近的股票交易日的股价曲线斜率
        | 斜率s                        | 标签                      |
        | ---------------------------- | ------------------------- |
        | s < -THRESH_VERY             | very_negative（卖出）     |
        | -THERESH_VERY <= s < -THRESH | negative（适度减持）      |
        | -THRESH <= s <= THRESH       | neutral（观望）           |
        | THRESH < s <= THRESH_VERY    | positive（买入）          |
        | s > THRESH_VERY              | very_positive（强力买入） |
    * 添加新闻/公告发布前FORWARD_DAYS天的股价曲线斜率作为判断信息
5. 运行`arrange_dataset.py`打包同日新闻信息，数据存储于`fin_data/content_with_labels`
    * 目前设置的打包方式为：若某只股票在同一天的新闻+公告数大于5，则按5条一组的方式随机打包为多组
    * 公告信息使用“【公告】”进行了额外标记
6. 运行`making_dataset.py`将数据集处理为jsonl格式，数据集存储于`fin_data/jsonl`
    * train_start_date到test_start_date之间的数据设置为训练集，test_start_date之后的设置为测试集
    * 最后会打印出训练集大小、测试集大小和最大数据长度，在进行p-tuning的时候`max_source_length`不应小于这个长度

## ChatGLM2-6B p-tuning
1. git clone [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)的代码到本地，安装对应的依赖。
2. 可以选择去[huggingface](https://huggingface.co/THUDM/chatglm2-6b)下载ChatGLM2的参数，可能会比在运行时由代码自动从huggingface上拉取更快
3. 进入ChatGLM2-6B代码的`ptuning`文件夹，运行`train.sh`，运行前需要修改如下参数：
    * train_file和validation_file指向之前处理的jsonl文件
    * prompt_column设为instruction
    * response_column设为target
    * model_name_or_path设置为下载的ChatGLM2参数的位置
    * output_dir设为保存p-tuning参数的位置
    * max_source_length设为之前运行`making_dataset.py`
    * quantization_bit设为8
4. 训练后可通过`evaluate.sh`进行测试，我将其修改如下，可参考：
    ```python
    PRE_SEQ_LEN=128
    NUM_GPUS=1

    for ((STEP = 100; STEP <= 3000; STEP += 100));
    do
        torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
            --do_predict \
            --validation_file /home/hupx/baiyuzhuo/finnlp-data/fin_data/jsonl/title_test_slope.json \
            --test_file /home/hupx/baiyuzhuo/finnlp-data/fin_data/jsonl/title_test_slope.json \
            --overwrite_cache \
            --prompt_column instruction \
            --response_column target \
            --model_name_or_path /home/hupx/baiyuzhuo/finnlp-data/chatglm-model/chatglm2-6b \
            --ptuning_checkpoint /home/hupx/baiyuzhuo/finnlp-data/chatglm-model/fin_model/slope_chatglm2-6b-pt-128-2e-2/checkpoint-$STEP \
            --output_dir /home/hupx/baiyuzhuo/finnlp-data/chatglm-model/fin_model/evaluate_slope_result/checkpoint-$STEP \
            --overwrite_output_dir \
            --max_source_length 460 \
            --max_target_length 8 \
            --per_device_eval_batch_size 1 \
            --predict_with_generate \
            --pre_seq_len $PRE_SEQ_LEN \
            --quantization_bit 8
    done
    ```
5. 可通过运行`evaluation/calculate_accuracy.py`计算准确率，`evaluation/calculate_label_num.py`计算各标签下实际标签分布，运行前记得修改对应的文件路径；也可以通过ChatGLM2-6b/ptuning文件夹中的`web_demo.py`在网页上直接进行生成