import re
import json
import sys, os

def single_sentence_cleanup(text, response):
    # Split by period and take the first sentence
    
    if response in text:
        return text.split(response)[1]
    else:
        return text
    

if __name__=='__main__':
    #decoder 모델의 생성 결과의 후처리 (최종 output 부분 이외의 부분 제거)

    # 다시 후처리 함수와 jsonl 파일 처리를 실행합니다.
    filename = sys.argv[1]
    # response = sys.argv[2]
    response = '### 응답:\\n'

    # 제공된 jsonl 파일을 읽기
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 각 데이터에 대해 후처리 적용
    for item in data:
        item["output"] = single_sentence_cleanup(item["output"], response)
        if '문장1: ' in item['output']:
            item['output']= item['output'].replace('문장1: ', '')
        if '문장2: ' in item['output']:
            item['output']= item['output'].replace('문장2: ', '')


    # 결과를 다시 jsonl 파일로 저장
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")