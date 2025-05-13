import os, json, re

txt_file   = "./dialog_04_scripts.txt"
output_dir = "./script"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    # ① (A)/(B)  → A  ――  (닫힘)/열림 구조)
    text = re.sub(r'\(([^)]+)\)\s*/\s*\(([^)]+)\)', r'\1', text)

    # ② (A/B)    → A  ――  한 쌍의 괄호 안에 / 존재
    text = re.sub(r'\(([^)]+?)\s*/\s*([^)]+?)\)', r'\1', text)

    # ③ 주석 기호(b/, l/, o/, u/, n/) 제거
    text = re.sub(r'\b[bloun]/', '', text)

    # ④ 특수 기호 *, + 제거
    text = re.sub(r'[*+]', '', text)

    # ⑤ ‘단어/단어’   → 앞 단어만         (공백 유무 모두 대응)
    text = re.sub(r'\b(\S+)\s*/\s*\S+\b', r'\1', text)

    # ⑥ 중복 괄호·슬래시 정리
    text = re.sub(r'[()]', '', text)           # 남은 괄호 제거
    text = re.sub(r'\s*/\s*', ' ', text)       # 남은 / 제거

    # ⑦ 공백 정리
    return re.sub(r'\s+', ' ', text).strip()

# ------- JSON 생성 -------
with open(txt_file, encoding="utf-8") as f:
    for line in f:
        if "::" not in line:
            continue

        pcm_path, raw = map(str.strip, line.split("::", 1))
        wav_path = pcm_path.replace(".pcm", ".wav")
        file_id  = os.path.splitext(os.path.basename(wav_path))[0]

        # json_data = {
        #     "fileName": wav_path.replace("/", "\\"),
        #     "transcription": clean_text(raw)
        # }

        json_data = {
            "fileName": os.path.basename(wav_path),
            "transcription": clean_text(raw)
        }


        out = os.path.join(output_dir, f"{file_id}.json")
        with open(out, "w", encoding="utf-8") as g:
            json.dump(json_data, g, ensure_ascii=False, indent=2)
