import requests
from bs4 import BeautifulSoup
import csv

def fetch_categories_and_urls(url):
    # 发送HTTP请求获取网页内容
    response = requests.get(url)
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有的类别按钮
    buttons = soup.find_all('div', class_='category_btn')

    urls = []
    categories = []
    for button in buttons:
        # 每个按钮中的<a>标签包含所需的链接
        link = button.find('a')
        if link:
            urls.append(link['href'])
            categories.append(link.get_text(strip=True))  # 获取链接内的文本并清除空格

    return categories, urls

def fetch_riddles(url):
    # 发送HTTP请求获取网页内容
    response = requests.get(url)
    # 使用BeautifulSoup解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 假设每个谜语都在<p>标签中，具有一个特定的ID
    riddles = soup.find_all('p', id=lambda x: x and x.startswith('Riddle-'))
    
    data = []
    for riddle in riddles:
        if riddle.strong:
            # 试图提取谜语文本，检查是否存在后续兄弟节点
            riddle_text = riddle.find('strong', string=lambda x: 'Riddle:' in x)
            if riddle_text and riddle_text.next_sibling:
                riddle_text = riddle_text.next_sibling.strip()

            answer_tag = riddle.find('strong', string=lambda x: 'Answer:' in x)
            if answer_tag:
            # 对next_sibling进行更鲁棒的检查
                answer = answer_tag.next_sibling
                if answer and isinstance(answer, str):
                    answer = answer.strip()
                elif answer:
                # 处理嵌套的HTML元素
                    answer = answer.get_text(strip=True) if hasattr(answer, 'get_text') else None


            if riddle_text and answer:
                data.append([riddle_text, answer])
        if riddle.b:  # Changed from riddle.strong to riddle.b
            riddle_text = riddle.find('b', string=lambda x: 'Riddle:' in x)
            if riddle_text and riddle_text.next_sibling:
                riddle_text = riddle_text.next_sibling.strip()

            answer_tag = riddle.find('b', string=lambda x: 'Answer:' in x)
            if answer_tag:
            # 对next_sibling进行更鲁棒的检查
                answer = answer_tag.next_sibling
                if answer and isinstance(answer, str):
                    answer = answer.strip()
                elif answer:
                # 处理嵌套的HTML元素
                    answer = answer.get_text(strip=True) if hasattr(answer, 'get_text') else None

            if riddle_text and answer:
                data.append([riddle_text, answer])

    print(data)
    return data


def save_to_csv(main_category, data, filename):
    modified_data = [[main_category] + row for row in data]
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(modified_data)

if __name__ == '__main__':
    url = 'https://www.riddles.com/posts'  # 修改为正确的URL
    categories, category_urls = fetch_categories_and_urls(url)
    print("Categories:", categories)
    print("URLs:", category_urls)

    with open('data/riddles.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Main_category', 'Riddle', 'Answer'])

    for i in range(len(categories)):
        main_category, url = categories[i], category_urls[i]
        print(categories[i])
        riddles_data = fetch_riddles(url)
        save_to_csv(main_category, riddles_data, 'data/riddles.csv')
