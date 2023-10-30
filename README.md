# LocalNaturalFish

2023 국립국어원 인공지능 경진대회 모두의 말뭉치 SC 이야기 생성 '국내산 자연어'팀

###	모델 스펙

#### Backbone 모델 (사전학습 모델)


Pko-t5-large (800M): 나무위키, 위키피디아, 모두의 말뭉치 등을 사전학습한 t5 v1.1 모델
https://huggingface.co/paust/pko-t5-large

KULLM-Polyglot-12.8B-v2 (12.8B):  EleutherAI의 polyglot-ko-12.8b 사전학습 모델에 Low Rank Adaptation (LoRA)를 이용하여 GPT4ALL, Dolly, Vicuna 데이터셋 학습
https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2

```
├── clean_decoder_result.py
├── clean_decoder.sh
├── data
│   ├── dev.jsonl
│   ├── nikluge-sc-2023-test.jsonl
│   ├── sample2.jsonl
│   ├── sample.jsonl
│   ├── test.jsonl
│   ├── test_sample.jsonl
│   ├── train+dev.jsonl
│   └── train.jsonl
├── ensemble.py
├── ensemble.sh
├── environment.yaml
├── inference_korsts.py
├── inference.py
├── inference.sh
├── logfile.log
├── modules
│   ├── arg_parser.py
│   ├── dataset_preprocessor.py
│   ├── logger_module.py
│   ├── trainer.py
│   └── utils.py
├── README.md
├── results
├── run.py
├── submission
│   └── validation
├── train.sh
├── upload_model.py
└── upload_model.sh
```

#### 앙상블 모델 (Semantic Textual Similarity Voting)


daekeun-ml/koelectra-small-v3-korsts: 34GB의 한국어 text를 학습한 monologg/KoELECTRA의 small scale 모델을 KakaoBrain의 KorSTS 데이터셋으로 finetuning한 모델
https://huggingface.co/daekeun-ml/koelectra-small-v3-korsts

###	라이브러리 

	Python 3.9.0
	Transformers 4.33.0
	Pytorch 2.0.1
	PEFT 0.5.0

## SeGOESSi (Sentence Generation Output Ensembling by Semantic Similarity)

적용 방법론: 문장 생성 결과의 의미적 유사도 기반 앙상블 기법

일반적인 텍스트 생성 모델은 생성 결과가 하나의 완전한 문장임으로 일반적인 앙상블 기법을 사용할 수 없어, 현재까지 문장 생성 모델의 앙상블 기법은 아직까지 국내외로 알려진 바가 없습니다. 
본 국내산 자연어 팀은 이야기 생성 과제에서 단일 모델들이 아래와 같은 다양한 방법과 조건 환경에서 학습될 수 있고, 이때 각각 다른 생성결과가 나온 다는 점을 주목하였습니다.

![image](https://github.com/JuHyng/GCULocalNaturalFish/assets/90828283/326303aa-f0c5-41c5-9008-96e1f5877cfe)

각 개별 모델의 생성된 완성 문장들의 텍스트 의미적 유사도 (Semantic Textual Similarity)를 기반으로 보팅하는 방식을 고안하여 적용하였습니다. 

![image](https://github.com/JuHyng/GCULocalNaturalFish/assets/90828283/64ff4ae8-215f-45d1-ae66-6d9b3594ad68)



---

### 추론 process

추론 과정 요약
1. 데이터추가 및 conda 가상환경 설정
2. inference.sh 실행
3. clean_decoder.sh 실행
4. ensemble.sh 실행 

과정 상세 설명

1. 데이터 폴더 준비
   - data 폴더내에 업로드된 데이터파일을 추가합니다.
   - train.jsonl,val.jsonl,test.jsonl

2. 각 모델 별 추론
   - inference.sh 스크립트를 실행합니다.
   - 실행 시 최종출력에 필요한 모델 생성 결과들이 자동으로 일괄적으로 생성됩니다.
   - 제출용 파일 폴더 ./submission 아래 검증용 폴더 validation이 생성된 후, 아래와 같이 모델/체크포인트/생성 전략 별 생성 결과가 생성됩니다
   - ./submission
   - ㄴ validation
   -   ㄴ

3. 디코더 계열모델 생성결과 후처리
   - clean_decoder.sh 스크립트를 실행합니다.
   - Auto-regressivie decoder 계열 특성 상 프롬프트 템플릿 뒤에 붙혀 생성되어 이를 제거하도록 스크립트를 작성하였습니다.
   - 디코더 모델 출력의 '### 응답:\\n' 이전의 문자열이 제거됩니다.

4. 앙상블 진행
   - ensemble.sh 스크립트를 실행합니다.
   - 추론할 파일 ./data/test.jsonl에 대해 검증용 폴더 ./submission/validation 하위에 있는 생성 결과들로 앙상블 voting이 진행됩니다/
   - 최종 출력파일 ./submission/validation/최종꼬시.jsonl 파일이 생성됩니다.
