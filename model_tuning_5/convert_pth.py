
import torch
from transformers import Wav2Vec2ForCTC

# 1) 파인튜닝된 모델 폴더 경로
MODEL_DIR = "./wav2vec2_korean_v7"

# 2) .pth로 저장할 파일명
PTH_PATH  = "./wav2vec2_korean_v7_pth_ver.pth"

def main():
    # 모델 불러오기 (config 포함)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)

    # state_dict만 저장 ─ 구조는 따로 로드 필요
    torch.save(model.state_dict(), PTH_PATH)

    print(f"✅ 저장 완료: {PTH_PATH}")

if __name__ == "__main__":
    main()
