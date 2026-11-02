import re
import pandas as pd


def split_text_to_sentences(text):
    """
    按中文标点分句：支持。！？；四种分句符号
    处理连续标点、空行等情况
    """
    # 定义分句正则表达式（匹配分句标点，保留标点在原句末尾）
    sentence_pattern = r'([^。！？；]+[。！？；])'
    # 匹配所有句子
    sentences = re.findall(sentence_pattern, text.strip())
    # 过滤空句子（避免纯标点或空格的情况）
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    return sentences


def txt_to_excel(txt_path, excel_path):
    """
    读取txt文件，分句后导出为Excel
    :param txt_path: 输入txt文件路径（绝对路径）
    :param excel_path: 输出Excel文件绝对路径
    """
    try:
        # 读取txt文件（支持utf-8编码，若报错可改为gbk）
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # 分句处理
        sentences = split_text_to_sentences(text_content)

        if not sentences:
            print("警告：未识别到有效句子！")
            return

        # 创建编号和句子的列表（编号从1开始）
        data = {
            "句子编号": list(range(1, len(sentences) + 1)),
            "句子内容": sentences
        }

        # 创建DataFrame并保存为Excel
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False, engine='openpyxl')

        print(f"处理完成！")
        print(f"共识别到 {len(sentences)} 个句子")
        print(f"Excel文件已保存至：{excel_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_path}，请检查文件路径是否正确")
    except Exception as e:
        print(f"错误：处理过程中出现异常 - {str(e)}")


if __name__ == "__main__":
    # TXT文件绝对路径（用r开头表示原始字符串，避免反斜杠转义错误）
    TXT_FILE_PATH = r"C:\Users\user\Desktop\word\四川省委一号文件\2025四川省一号文件.txt"

    # Excel输出路径（保存到同一个文件夹，文件名带“分句结果”后缀）
    EXCEL_FILE_PATH = r"C:\Users\user\Desktop\word\四川省委一号文件\2025四川省一号文件_分句结果.xlsx"

    # 调用函数，生成Excel
    txt_to_excel(TXT_FILE_PATH, EXCEL_FILE_PATH)