# base-detection

## 🔄 CI

```bash
git remote add upstream https://github.com/bert13069598/base-train.git
```

## 📋 dependency

> python 3.12

```bash
uv pip install -r requirements.txt
```

## 📂 prepare dataset

> python loader.py \<idx> <show | make>

예시

```bash
python loader.py 0 --show
```

```bash
python loader.py 0 --make yolo
```

- 인덱스 : 준비할 데이터셋 idx
- `--init` : 데이터셋 생성
- `--show` : 라벨링 이미지 확인
- `--make` : 학습 데이터셋 저장 (`yolo`, `coco`)
- `--work` : make할 때 멀티프로세스 코어 갯수
- `--path` : 학습 데이터 저장 위치 (기본 위치 : `yaml`의 `path`)

1. 데이터셋 생성

```bash
python loader.py --init <데이터셋명>
```

1. [cfg/datasets](cfg/datasets)/<데이터셋명>.yaml 작성
2. [dataloader/loader](dataloader/loader)/loader_<데이터셋명>.py 작성
3. `loader.py` 옵션 지정 후 구동

| dataset | folder path / filename | idx | total | link |
|---------|------------------------|:---:|-------|------|
|         |                        |     |       |      |

## 🚀 train dataset

<데이터셋명>과 동일하게 <프로젝트> 설정

> python train.py -m <모델명> <-o> -p <프로젝트>

예시

```bash
python train.py -m yolov8s -p car
```

1. [cfg/datasets](cfg/datasets)/<데이터셋명>.yaml 경로 확인
2. `train.py` 구동

### yolo dataset structure

```
.
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

### tensorboard

```bash
tensorboard --logdir runs/yolov8s/car
```

## ✨ export model

> python export.py -m <모델명> <-o> -p <프로젝트> -b <배치 수>

예시

```bash
python export.py -m yolov8s -p car -b 1
```

`pt` -> `onnx`

현재 yolo만 지원

## 🎯 check result

> python test.py -m <모델명> <-o> -p <프로젝트> --show --auto

예시

```bash
python test.py -m yolov8s -o -p car --show
```

```bash
python test.py -m yolov8s -o -p car --auto
```

- `--show` : 추론 결과 확인
- `--auto` : 오토라벨링
- `--dirs` : 데이터 로드 위치 (기본 위치 : `yaml`의 `test`)

## useful command

`jpg` 전부 삭제

```bash
find . -type f -name "*.jpg" | xargs rm -f
```

`txt` 전부 삭제

```bash
find . -type f -name "*.txt" | xargs rm -f
```

`val` 폴더 안의 랜덤 파일 30개 `test` 폴더로 복사

```bash
find val -maxdepth 1 -type f | shuf -n 30 | xargs -I{} cp {} test/
```
