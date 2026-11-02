import os

# ================= 配置区域 =================
# 请将此处的路径修改为您电脑上 "word" 文件夹的实际绝对路径
# 例如：r"C:\Users\Administrator\Desktop\word"
base_dir = r"C:\Users\user\Desktop\word"

# 修改为 2004 到 2026 年 (range 的右边界是不包含的，所以写 2027)
years = range(2004, 2027)

# 定义不同文件的存放规则字典
file_configs = {
    "中央": {
        "folder": "农业农村部一号文件",
        "name_template": "{year}中央一号文件.txt"
    },
    "四川": {
        "folder": "四川省委一号文件",
        "name_template": "{year}四川省一号文件.txt"
    },
    "江苏": {
        "folder": "江苏省委一号文件",
        "name_template": "{year}江苏省委一号文件.txt"
    }
}


# ============================================

def read_file_content(file_path):
    """尝试用不同编码读取文件内容，返回字符数（剔除空白和换行）"""
    if not os.path.exists(file_path):
        return 0, False

    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                # 统计纯字数，去除空格和换行符
                clean_content = content.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                return len(clean_content), True
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件出错 {file_path}: {e}")
            return 0, False

    print(f"警告：无法解析文件编码 {file_path}")
    return 0, False


def main():
    print(f"开始统计 {years[0]}-{years[-1]} 年的一号文件字数...\n")
    print("-" * 40)

    total_all = 0

    for region, config in file_configs.items():
        region_total = 0
        print(f"【{region}一号文件】:")

        for year in years:
            folder_path = os.path.join(base_dir, config["folder"])
            file_name = config["name_template"].format(year=year)
            file_path = os.path.join(folder_path, file_name)

            char_count, exists = read_file_content(file_path)

            if exists:
                print(f"  - {year}年: {char_count} 字")
                region_total += char_count
            else:
                print(f"  - {year}年: [文件不存在，跳过]")

        print(f"  >>> {region} {years[0]}-{years[-1]} 年总字数: {region_total} 字\n")
        total_all += region_total

    print("-" * 40)
    print(f"所有文件合计总字数: {total_all} 字")


if __name__ == "__main__":
    main()