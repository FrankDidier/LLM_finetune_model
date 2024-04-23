import os
import json
import csv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_url, cached_download
import pandas as pd


instruction_prompt = \
"""
给你一篇文章的主题。 创建一篇引人入胜的文章，对所选主题提供独特的视角。 首先制作一个引人注目的标题，反映内容的本质。
将文章概述为引言、正文和结论部分。 以强有力的开场白开始，深入研究正文的详细分析，最后总结要点。
在适当的情况下加入轶事或视觉效果以确保参与度。 修改您的文章，使其清晰、连贯和正确。 最后以号召性用语来鼓励读者参与。
旨在发表经过充分研究、富有洞察力和原创性的评论，符合新闻标准并引发有意义的讨论。
"""
# English translation
# You are given the topic of an article. Create a captivating article that offers a unique perspective on a chosen topic. Start by crafting a compelling title that reflects the essence of your content. Outline your article into introduction, body, and conclusion sections. Begin with a strong opening statement, delve into detailed analysis in the body, and conclude by summarizing key points. Ensure engagement by incorporating anecdotes or visuals where appropriate. Revise your article for clarity, coherence, and correctness. End with a call to action to encourage reader participation. Aim for a well-researched , insightful, and original commentary that adheres to journalistic standards and sparks meaningful discussion.

prompt_template = \
"""
命令: {}

话题: {}

输出文字: 
"""


def read_json_to_csv(json_folder, output_csv):
    """
    Reads 'url', 'query_message', and 'article_title' fields from JSON files in a folder
    and saves them to a CSV file.

    Args:
      json_folder (str): Path to the folder containing JSON files.
      output_csv (str): Path to the output CSV file.
    """

    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['url', 'output', 'instruction'])

            # Loop through JSON files in the folder
            for filename in tqdm(os.listdir(json_folder)):
                if filename.endswith('.json'):  # Check for JSON extension
                    json_path = os.path.join(json_folder, filename)

                    try:
                        with open(json_path, 'r') as json_file:
                            data = json.load(json_file)

                            try:
                                url = data['url'].split(' ')[0]
                                output = data['query_message']
                                topic = data['article_title']
                                instruction = prompt_template.format(instruction_prompt, topic)
                                writer.writerow([url, output, instruction])
                            except KeyError as e:
                                print(f"Error: Missing key(s) in {filename}: {e}")
                    except FileNotFoundError:
                        print(f"Error: JSON file not found: {json_path}")

    except PermissionError as e:
        print(f"Error: Permission denied while writing to CSV: {e}")


def create_and_upload_hf_dataset(csv_path, push_to_hub=False, dataset_url=None, private=False):
    """
    Creates a Hugging Face dataset from a CSV file and uploads it to the Hub.

    Args:
        csv_path (str): Path to the CSV file.
    """

    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    if push_to_hub:
        dataset.push_to_hub("DavideTHU/chinese_news_dataset", private=private)

    if dataset_url:
        print(f"Dataset uploaded to Hugging Face Hub: {dataset_url}")

    return dataset


if __name__ == '__main__':
    json_folder = 'raw_data'
    output_csv = 'preprocessed_data/data.csv'
    dataset_url = 'https://huggingface.co/datasets/DavideTHU/chinese_news_dataset'
    push_to_hub = False
    private = False
    #read_json_to_csv(json_folder, output_csv)
    dataset = create_and_upload_hf_dataset(output_csv, push_to_hub=push_to_hub, dataset_url=dataset_url, private=private)
    print(dataset)

