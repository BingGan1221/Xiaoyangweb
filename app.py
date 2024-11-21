import streamlit as st
import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import numpy as np
import os
import shutil
from datetime import datetime
from pathlib import Path
import requests

# 设置页面配置
st.set_page_config(
    page_title="Excel评论分析工具",
    page_icon="📊",
    layout="wide"
)

# 添加自定义CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .comment-box {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        background-color: white;
    }
    .stats-box {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 定义停用词
@st.cache_data
def get_stop_words():
    return {
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '之', '用', '于', '把', '等', '去', '又', '能', '好', '在',
        '或', '这', '那', '有', '很', '只', '些', '为', '呢', '啊',
        '并', '给', '跟', '还', '个', '之类', '各种', '没有', '非常',
        '可以', '因为', '因此', '所以', '但是', '但', '然后', '如果',
        '虽然', '这样', '这些', '那些', '如此', '只是', '真的', '一个',
        '不过', '这个', '那个', '它们', '他们', '她们', '咱们', '您们',
        '一些', '一样', '一般', '一直', '而且', '而是', '而言', '而已',
        '出来', '过来', '起来', '下来', '上来', '正在', '开始', '一下',
        '这种', '那种', '有些', '有点', '比较', '越来越', '只要', '只有',
        '大约', '大概', '大家', '如何', '什么', '哪些', '对于', '这么',
        '那么', '几乎', '差不多', '恐怕', '应该', '没什么', '看看', '按照',
        '除了', '除此之外', '所有', '大多', '许多', '然而', '不仅', '不但',
        '总之', '总的来说', '总的来看', '总的说来', '总而言之', '相对而言',
        '具体', '具体地说', '具体说来', '综上所述', '简言之', '一般来说',
        '一般说来', '实际上', '事实上', '当然', '应该说', '确切地说',
        '现在', '如今', '一直', '始终', '一段时间', '总是', '曾经',
        '已经', '已', '曾', '正在', '将要', '将来', '最后', '最终',
        '可能', '也许', '大概', '或许', '似乎', '好像', '看起来', '看来',
        '看样子', '好象', '可能', '没准', '估计', '想必', '大约', '至少',
        '起码', '不至于', '多少', '恐怕', '没想到', '难道', '究竟', '到底',
        '为什么', '怎么', '怎么样', '什么样', '哪样', '如何', '咋', '何',
        '何时', '何处', '如此', '怎样', '怎么办', '怎么样', '这么样',
        '那么样', '多么', '这么', '那么', '真是', '真的', '确实', '实在',
        '难怪', '怪不得', '居然', '竟然', '竟', '实在', '简直', '其实',
        '其实', '究竟', '没想到', '大概', '不知道', '不知', '不会', '不能',
        '不可能', '不可以', '不要', '不必', '不一定', '不一样', '不同',
        '不如', '不成', '不拘', '不管', '不论', '不仅', '不但', '不只',
        '不外乎', '不止', '不够', '不了', '不得', '不得不', '不得了',
        '不得已', '不至于', '不如', '不若', '不足', '不过', '不限于',
        '不下', '不光', '不单', '不独', '不料', '不免', '不必', '不怕',
        '不惟', '不问', '不比', '不是', '不错', '不常', '不至于', '不能不',
        '不成', '不拘', '不特', '不独', '不管', '不论', '不仅', '不但',
        '不只', '不外乎', '不止', '不够', '不了', '不得', '不得不', '不得了',
    }

# 处理Excel文件
def process_excel(uploaded_file):
    try:
        # 读取Excel文件
        wb = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # 查找总安排打分列
        score_col = None
        for col in wb.columns:
            if '总安排打分' in str(col):
                score_col = col
                break
        
        if score_col is None:
            st.error('未找到"总安排打分"列')
            return None
            
        return wb.iloc[:, 0], wb[score_col]  # 返回评论列和分数列
        
    except Exception as e:
        st.error(f'处理Excel文件时出错: {str(e)}')
        return None

# 获取字体路径
def get_font_path():
    # 首先检查项目的 fonts 目录
    font_paths = [
        Path(__file__).parent / 'fonts' / 'simhei.ttf',  # 项目fonts目录
        Path('C:/Windows/Fonts/simhei.ttf'),  # Windows系统字体目录
        Path('/usr/share/fonts/truetype/simhei.ttf'),  # Linux系统字体目录
        Path('/System/Library/Fonts/simhei.ttf'),  # macOS系统字体目录
    ]
    
    for font_path in font_paths:
        if font_path.exists():
            return str(font_path)
    
    # 如果找不到字体，使用默认无衬线字体
    return None

def ensure_font():
    """确保中文字体文件存在"""
    font_dir = Path('fonts')
    font_path = font_dir / 'simhei.ttf'
    
    # 如果字体文件已存在，直接返回路径
    if font_path.exists():
        return str(font_path)
    
    # 创建fonts目录
    font_dir.mkdir(exist_ok=True)
    
    # 字体文件的下载链接（这里使用一个可靠的源）
    font_url = "https://github.com/microsoft/Windows-Font/raw/master/SimHei.ttf"
    
    try:
        # 下载字体文件
        response = requests.get(font_url)
        response.raise_for_status()
        
        # 保存字体文件
        with open(font_path, 'wb') as f:
            f.write(response.content)
        
        return str(font_path)
        
    except Exception as e:
        st.error(f"下载字体文件失败: {str(e)}")
        return None

def main():
    st.title("Excel评论分析工具")
    
    # 确保字体文件存在
    font_path = ensure_font()
    if not font_path:
        st.error("无法加载中文字体文件，词云图可能无法正确显示中文。")
    
    uploaded_file = st.file_uploader("上传Excel文件", type=['xlsx'])
    
    if uploaded_file:
        result = process_excel(uploaded_file)
        
        if result is not None:
            comments, scores = result
            
            # 显示基本统计信息
            total_comments = len(comments)
            low_score_comments = sum(1 for score in scores if score <= 3)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总评论数", total_comments)
            with col2:
                st.metric("差评数量", low_score_comments)
            
            # 分词和统计
            stop_words = get_stop_words()
            word_freq = Counter()
            word_freq_low = Counter()
            word_comments = {}
            word_comments_low = {}
            
            for comment, score in zip(comments, scores):
                if pd.isna(comment) or pd.isna(score):
                    continue
                    
                comment = str(comment).strip()
                if not comment:
                    continue
                
                words = jieba.cut(comment)
                for word in words:
                    word = word.strip()
                    if (len(word) > 1 and
                        word not in stop_words and
                        not word.isdigit()):
                        
                        word_freq[word] += 1
                        if word not in word_comments:
                            word_comments[word] = []
                        word_comments[word].append(comment)
                        
                        if score <= 3:
                            word_freq_low[word] += 1
                            if word not in word_comments_low:
                                word_comments_low[word] = []
                            word_comments_low[word].append(comment)
            
            # 创建标签页
            tab1, tab2 = st.tabs(["所有评论分析", "差评分析"])
            
            # 所有评论分析
            with tab1:
                if word_freq:
                    generate_analysis(word_freq, word_comments, "所有评论")
                else:
                    st.info("没有找到有效的评论数据")
            
            # 差评分析
            with tab2:
                if word_freq_low:
                    generate_analysis(word_freq_low, word_comments_low, "差评")
                else:
                    st.info("没有找到差评数据")

def generate_analysis(word_freq, word_comments, title):
    st.subheader(f"{title}词云图")
    
    # 获取字体路径
    font_path = get_font_path()
    
    # 生成词云图
    try:
        wc = WordCloud(
            font_path=font_path,  # 使用获取到的字体路径
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            # 如果没有找到中文字体，使用默认设置
            font_path=font_path if font_path else None,
        )
        wc.generate_from_frequencies(word_freq)
        
        # 显示词云图
        st.image(wc.to_array())
        
    except Exception as e:
        st.error(f"生成词云图时出错: {str(e)}")
        st.info("提示：可能是因为缺少中文字体文件，请确保系统中安装了中文字体。")
    
    # 生成词频柱状图
    st.subheader(f"{title}词频统计")
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
    
    fig = px.bar(
        x=list(top_words.keys()),
        y=list(top_words.values()),
        labels={'x': '词语', 'y': '出现次数'},
        title=f'词频统计（前20个词）'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加词语选择器
    selected_word = st.selectbox(
        "选择词语查看相关评论",
        options=list(top_words.keys())
    )
    
    if selected_word:
        st.subheader(f"包含 '{selected_word}' 的评论")
        for comment in word_comments[selected_word]:
            st.markdown(f'<div class="comment-box">{comment}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()