import streamlit as st
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image
from ExactFeature import FeatureExtractor

SIDEBAR_OPTIONS = ["项目信息", "上传图片", "使用预置图片"] # 设置下拉列表可选项

fe = FeatureExtractor()
features = []
img_paths = []

# 读取图库数据
for feature_path in Path("./database/feature").glob("*.npy"):
    features.append(np.load(str(feature_path)))
    img_paths.append(Path("./database/image") / (feature_path.stem + ".jpg"))
features = np.array(features)

# 计算图片特征相似度
def cosine_similarity(f1, f2):
    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    cos_sim = dot / (norm1 * norm2)
    return cos_sim

#  加载系统说明
def get_file_content_as_string(path):
    # url = 'https://gitee.com/wu_jia_sheng/graduation_program/blob/master/' + path
    url = 'https://raw.githubusercontent.com/Alex0Stephen/Image_Retrieval/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

# 展示搜索结果
def display_result(imgs_list):
    st.title("搜索结果(仅显示相似度大于0.5)：")
    for img_msg in imgs_list:
        if img_msg[0] > 0.5:
            img = Image.open(img_msg[1])
            st.image(img, caption="相似度：" + str(img_msg[0]))


if __name__ == '__main__':

    st.set_page_config(page_title="Welcome To Image", page_icon=":rainbow:")

    st.sidebar.warning('请上传图片')
    st.sidebar.write(" ------ ")
    st.sidebar.title("让我们来一起探索吧")

    app_mode = st.sidebar.selectbox("请从下列选项中选择您想要的功能", SIDEBAR_OPTIONS)

    st.title('Welcome To Image Search System')
    st.write(" ------ ")

    # 根据不同选择实现不同功能
    if app_mode == "项目信息":
        st.sidebar.write(" ------ ")
        st.sidebar.success("项目信息请往右看!")
        st.write(get_file_content_as_string("Project-Info.md"))
        # st.write(Path("./Project-Info.md"))

    elif app_mode == "上传图片":
        st.sidebar.write(" ------ ")
        file = st.file_uploader('上传图片', type=["png", "jpg", "jpeg"])
        if file is not None:
            img = Image.open(file)

            st.title("上传图片为：")
            image = img.resize((224, 224))
            st.image(image)

            pressed = st.sidebar.button('搜索')

            if pressed:
                st.empty()
                st.sidebar.write('请稍等! 你知道的，这通常需要一点时间。')

                query = fe.extract(img)

                # 计算图片与图库中图像数据相似度，并从数值高度到底进行排序
                scores = [(cosine_similarity(query, features[index]), img_paths[index]) for index in range(len(img_paths))]
                scores.sort(key=lambda x: x[0], reverse=True)
                display_result(scores)

        # else:
        #     st.warning("上传图片失败!")

    elif app_mode == "使用预置图片":
        st.sidebar.write(" ------ ")

        st.title("预置图片为：")
        img_fixed = Image.open(Path("./database/image/101400.jpg")).resize((224, 224))
        st.image(img_fixed)

        pressed = st.sidebar.button('搜索')

        if pressed:
            st.empty()
            st.sidebar.write('请稍等! 你知道的，这通常需要一点时间。')

            query = fe.extract(img_fixed)

            scores = [(cosine_similarity(query, features[index]), img_paths[index]) for index in
                          range(len(img_paths))]
            scores.sort(key=lambda x: x[0], reverse=True)
            display_result(scores)


