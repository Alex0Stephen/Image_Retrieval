from ExactFeature import FeatureExtractor
from pathlib import Path
import numpy as np
from PIL import Image

if __name__ == '__main__':
    fe = FeatureExtractor()

    # 读取图库图片并提取特征
    for img_path in sorted(Path("./database/image").glob("*.jpg")):
        print(img_path)
        img = Image.open(img_path)

        # 提取特征
        feature = fe.extract(img)

        # 将图片特征进行保存
        feature_path = Path("./database/feature") / (img_path.stem + ".npy")
        np.save(str(feature_path), feature)
