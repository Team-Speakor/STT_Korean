import os, json, re

txt_file   = "./dialog_04_scripts.txt"
output_dir = "./script"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    # ① (A)/(B) 구조 → A 로 변환 (괄호/괄호 구조)
    text = re.sub(r'\(([^)]+)\)\s*/\s*\(([^)]+)\)', r'\1', text)

    # # ② (A/B) 구조 → A 로 변환 (괄호 내부의 / 구조)
    # text = re.sub(r'\(([^)/]+)/([^)/]+)\)', r'\1', text)

    # ③ 주석 기호 제거 (b/, l/, o/, u/, n/)
    text = re.sub(r'\b[bloun]/', '', text)

    # ④ 특수기호 제거 (*, +)
    text = re.sub(r'[*+]', '', text)

    # ⑤ 괄호 제거 (이 시점에서 괄호는 정제됐기 때문에 제거 안전)
    text = re.sub(r'[()]', '', text)

    # # ✅ ⑥ 마지막에 ‘단어/단어’ 패턴 제거 → 이 시점에서만 처리해야 안전
    # text = re.sub(r'\b(\S+)\s*/\s*\S+\b', r'\1', text)

    # ⑦ 슬래시 / 제거
    text = re.sub(r'\s*/\s*', ' ', text)

    # ⑧ 공백 정리
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
