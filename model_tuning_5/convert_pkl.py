
"""
wav2vec2_korean_v7 → wav2vec2_korean_v7.pkl 변환 스크립트
실행 전:
  pip install transformers torch --upgrade
"""

import torch
from transformers import Wav2Vec2ForCTC

# 1) 파인튜닝된 모델 폴더 경로
MODEL_DIR = "./wav2vec2_korean_v7"

# 2) .pkl로 저장할 파일명
PKL_PATH  = "./wav2vec2_korean_v7_pkl_ver.pkl"

def main():
    # 모델 불러오기 (config 포함)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)

    # state_dict만 저장 ─ 원본 폴더는 그대로 유지
    torch.save(model.state_dict(), PKL_PATH)

    print(f"✅  저장 완료: {PKL_PATH}")

if __name__ == "__main__":
    main()
