
charater-llm 의 논문을 참조하여 사용한 프롬프트로 데이터 생성
생성한 데이터로 unsloth 를 사용하여 파인튜닝하였습니다

finetuning.py
해당 파일로 파인튜닝합니다. unsloth 사용

infer.py 
fastapi 를 활용하여 클라우드 환경에서 로컬로 전송
클라우드에서는 chainlit 을 사용할 수 없어서 이 방법 사용

chainlit_app.py
로컬에서 챗봇 생성

참조 자료: 
https://arxiv.org/abs/2310.10158
https://github.com/unslothai/unsloth
